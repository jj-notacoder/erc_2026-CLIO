# Original method extracted byte-for-byte (apart from newline normalization).
# Source: certified collision_quality_final_08/source/erc_phase1_solution/manipulation_node.py
# Original node SHA256: 3b725aa86277950258c82ddeff87f295c344c605274ae9ff164cf39079af6524
# Final08 certificate SHA256: 79995071b49f46d980c42bbd28dbba0454d2053197cd04bb45eb8ff41e67f6e0
# This fixture contains only the original retained-send method, not a copied node.
from __future__ import annotations

class Baseline:
    def _send_retained_arm_trajectory(
        self,
        goal: FollowJointTrajectory.Goal,
        total_duration: float,
        legs: Sequence[Tuple[Sequence[float], float, str]],
        command: str,
        *,
        leg_offset: int = 0,
        initial_pressure_gate=None,
    ) -> Tuple[bool, bool]:
        """Send a continuous carried route and cancel it on payload danger."""
        if self._cancel.is_set():
            return False, False

        endpoint_contact_age = min(
            0.15,
            float(getattr(self, 'grasp_contact_max_age', 0.15)),
        )

        def watchdog_fault() -> Optional[str]:
            return self._payload_hazard_reason(max_age=endpoint_contact_age)

        def publish_fault(reason: str, started_ns: int) -> None:
            now_ns = self.get_clock().now().nanoseconds
            elapsed = max(0.0, (now_ns - started_ns) / 1e9)
            leg, phase = self._trajectory_leg_at_elapsed_time(legs, elapsed)
            self._publish_status(
                'grasp_lost',
                command=command,
                phase=phase,
                leg=leg + leg_offset,
                reason=reason,
            )

        # These constant-time gates catch a collision/contact loss that arrived
        # during the final probe callback itself without adding a planning dwell.
        initial_fault = watchdog_fault()
        if initial_fault is not None:
            publish_fault(initial_fault, self.get_clock().now().nanoseconds)
            return False, True

        acceptance_future = None
        acceptance_fault = None
        acceptance_token = object()
        def request_goal():
            # The pressure gate already holds the sensor mutex at admission.
            # Ordinary callers register under that mutex immediately below.
            pending = getattr(self, '_pending_retained_acceptances', set())
            pending.add(acceptance_token)
            self._pending_retained_acceptances = pending
            return self.arm_client.send_goal_async(goal)
        try:
            if initial_pressure_gate is not None:
                acceptance_future = initial_pressure_gate.send(request_goal)
            else:
                with self._lock:
                    acceptance_future = request_goal()
            acceptance_deadline = time.monotonic() + self.timeout
            while not acceptance_future.done():
                # A hazard may be transient while the action server accepts the
                # request. Remember it so a later fresh callback cannot authorize
                # an action that already lost its payload contract.
                acceptance_fault = acceptance_fault or watchdog_fault()
                if time.monotonic() >= acceptance_deadline:
                    raise TimeoutError('retained arm goal acceptance timed out')
                time.sleep(0.02)
            goal_handle = acceptance_future.result()
            if goal_handle is None or not isinstance(goal_handle.accepted, bool):
                raise RuntimeError('retained arm goal acceptance result is invalid')
        except LiftPressureRejected:
            # Rejected before publication: no recovery route or gripper-open
            # command is authorized by this unverified loaded state.
            raise
        except Exception as exc:
            self._cancel.set()
            if acceptance_future is not None:
                # The server might still start this request. Observe/cancel a
                # late handle without blocking this callback or guessing that a
                # timeout means the robot stopped.
                try:
                    acceptance_future.add_done_callback(
                        lambda future: self._cancel_late_retained_goal(
                            future, acceptance_token))
                except Exception:
                    pass
            raise RetainedMotionNotStopped(
                'retained arm goal acceptance has no confirmed terminal state'
            ) from exc
        if not goal_handle.accepted:
            with self._lock:
                self._pending_retained_acceptances.discard(acceptance_token)
            return False, False
        with self._lock:
            self._goal_handles.append(goal_handle)
            self._pending_retained_acceptances.discard(acceptance_token)
        keep_goal_registered = True
        try:
            started_ns = self.get_clock().now().nanoseconds
            result_future = goal_handle.get_result_async()
            if acceptance_fault is not None:
                publish_fault(acceptance_fault, started_ns)
                if not self._cancel_retained_goal_and_confirm(goal_handle, result_future):
                    self._cancel.set()
                    raise RetainedMotionNotStopped(
                        'unsafe accepted arm trajectory could not be cancelled'
                    )
                keep_goal_registered = False
                return False, True
            # Gazebo commonly runs below real time on the full TIAGo world.
            # Match the simulator-aware waits elsewhere in this node instead of
            # assuming one simulated trajectory second is one wall second.
            wall_deadline = (
                time.monotonic() + self.timeout + 4.0 * total_duration
            )
            while not result_future.done():
                if self._cancel.is_set():
                    if not self._cancel_retained_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        raise RetainedMotionNotStopped(
                            'retained arm trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    return False, False
                fault = watchdog_fault()
                if fault is not None:
                    publish_fault(fault, started_ns)
                    if not self._cancel_retained_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        self._cancel.set()
                        raise RetainedMotionNotStopped(
                            'unsafe retained arm trajectory could not be cancelled'
                        )
                    keep_goal_registered = False
                    return False, True
                if time.monotonic() >= wall_deadline:
                    if not self._cancel_retained_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        self._cancel.set()
                        raise RetainedMotionNotStopped(
                            'timed-out retained arm trajectory could not be cancelled'
                        )
                    keep_goal_registered = False
                    raise TimeoutError('retained arm trajectory timed out')
                time.sleep(0.02)
            wrapped = result_future.result()
            if not self._valid_retained_terminal_result(wrapped):
                raise RuntimeError('retained arm result has no valid terminal status')
            if self._cancel.is_set():
                # Cancellation can race an already terminal result, bypassing
                # the polling loop. A stopped action must not advance the route.
                keep_goal_registered = False
                return False, False
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                keep_goal_registered = False
                return False, False
            fault = watchdog_fault()
            if fault is not None:
                publish_fault(fault, started_ns)
                keep_goal_registered = False
                return False, True
            keep_goal_registered = False
            return True, False
        except RetainedMotionNotStopped:
            raise
        except Exception as exc:
            if keep_goal_registered:
                self._cancel.set()
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                raise RetainedMotionNotStopped(
                    'accepted retained arm action has no confirmed terminal state'
                ) from exc
            raise
        finally:
            if not keep_goal_registered:
                with self._lock:
                    if goal_handle in self._goal_handles:
                        self._goal_handles.remove(goal_handle)
