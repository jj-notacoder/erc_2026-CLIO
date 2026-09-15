"""Exact original R50/R51 _follow; complete source SHA 01a758d295ee2b4f79591821a6d03f8e53a633017366db74847f203823ba254d."""
from __future__ import annotations
class ManipulationNode:
    def _follow(
        self,
        client: ActionClient,
        names: Sequence[str],
        positions: Sequence[float],
        duration: float,
        *,
        head_preflight: bool = False,
        pre_send_check=None,
    ) -> bool:
        if self._cancel.is_set():
            return False
        _require_place_contact_clear(self)
        scene_reference = getattr(self, '_active_place_scene_reference', None)
        if scene_reference is not None:
            _require_place_contact_clear(self)
            measured_scene_context(self, scene_reference)
        with self._lock:
            payload_robot_contact = bool(
                getattr(self, '_payload_robot_watchdog_enabled', False)
                and getattr(self, '_target_robot_contact_latched', False)
            )
            if payload_robot_contact:
                self._payload_hazard_latched = 'payload_robot_contact'
        if payload_robot_contact:
            self._publish_status(
                'payload_hazard',
                reason='payload_robot_contact',
            )
            return False
        if head_preflight:
            if client is not self.head_client or tuple(names) != tuple(HEAD_JOINTS):
                raise ValueError('head preflight requires the head controller and joints')
            completed = getattr(self, '_last_completed_head_target', None)
            # A failed, cancelled or uncertain new action must not leave an
            # earlier success eligible for a later shortcut.
            self._last_completed_head_target = None
            held = self._try_completed_head_hold(positions, completed)
            if held is not None:
                return held
            if (self._held_book_corners is not None
                    and not self._carried_head_transition_is_safe(*positions)):
                raise RuntimeError('Carried book or robot blocks requested head motion')
        if not client.wait_for_server(timeout_sec=min(5.0, self.timeout)):
            raise RuntimeError(f'action server unavailable for {list(names)}')
        with self._lock:
            payload_robot_contact = bool(
                getattr(self, '_payload_robot_watchdog_enabled', False)
                and getattr(self, '_target_robot_contact_latched', False)
            )
            if payload_robot_contact:
                self._payload_hazard_latched = 'payload_robot_contact'
        if payload_robot_contact:
            self._publish_status(
                'payload_hazard',
                reason='payload_robot_contact',
            )
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = Duration(seconds=float(duration)).to_msg()
        goal.trajectory.points = [point]
        if scene_reference is not None:
            _require_place_contact_clear(self)
            measured_scene_context(self, scene_reference)
        if self._cancel.is_set():
            return False
        if getattr(self, '_place_contact_guard', None) is not None:
            # Serialize only this PLACE send against the contact-stop latch.
            # Do not hold the command lock while awaiting action acceptance.
            with self._adaptive_command_guard():
                _require_place_contact_clear(self)
                if self._cancel.is_set():
                    return False
                if pre_send_check is not None:
                    pre_send_check()
                if self._cancel.is_set():
                    return False
                acceptance = client.send_goal_async(goal)
            goal_handle = self._wait_future(acceptance, self.timeout)
        else:
            if pre_send_check is not None:
                pre_send_check()
            if self._cancel.is_set():
                return False
            goal_handle = self._wait_future(client.send_goal_async(goal), self.timeout)
        if goal_handle is None or not goal_handle.accepted:
            return False
        with self._lock:
            self._goal_handles.append(goal_handle)
        keep_goal_registered = True
        try:
            result_future = goal_handle.get_result_async()
            wall_deadline = time.monotonic() + self.timeout + 4.0 * duration
            while not result_future.done():
                if scene_reference is not None:
                    try:
                        _require_place_contact_clear(self)
                        measured_scene_context(self, scene_reference)
                    except RuntimeError:
                        if not self._cancel_goal_and_confirm(goal_handle, result_future):
                            keep_goal_registered = True
                            self._cancel.set()
                            raise RuntimeError('placement_scene_cancellation_unconfirmed')
                        keep_goal_registered = False
                        raise
                with self._lock:
                    payload_robot_contact = bool(
                        getattr(
                            self,
                            '_payload_robot_watchdog_enabled',
                            False,
                        )
                        and getattr(
                            self,
                            '_target_robot_contact_latched',
                            False,
                        )
                    )
                    if payload_robot_contact:
                        self._payload_hazard_latched = (
                            'payload_robot_contact'
                        )
                if payload_robot_contact:
                    self._publish_status(
                        'payload_hazard',
                        reason='payload_robot_contact',
                    )
                    if not self._cancel_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        raise RuntimeError(
                            'trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    return False
                if self._cancel.is_set():
                    if not self._cancel_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        raise RuntimeError(
                            'trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    return False
                if time.monotonic() >= wall_deadline:
                    if not self._cancel_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        self._cancel.set()
                        raise RuntimeError(
                            'trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    raise TimeoutError('trajectory timed out')
                time.sleep(0.02)
            # The result future and a contact callback can become ready in the
            # same executor turn.  Check the latched collision once more after
            # leaving the polling loop so a nominal controller success cannot
            # win that race.
            with self._lock:
                payload_robot_contact = bool(
                    getattr(self, '_payload_robot_watchdog_enabled', False)
                    and getattr(self, '_target_robot_contact_latched', False)
                )
                if payload_robot_contact:
                    self._payload_hazard_latched = 'payload_robot_contact'
            if payload_robot_contact:
                self._publish_status(
                    'payload_hazard',
                    reason='payload_robot_contact',
                )
                keep_goal_registered = False
                return False
            keep_goal_registered = False
            if self._cancel.is_set():
                return False
            if scene_reference is not None:
                _require_place_contact_clear(self)
                measured_scene_context(self, scene_reference)
            wrapped = result_future.result()
            keep_goal_registered = False
            succeeded = wrapped.status == GoalStatus.STATUS_SUCCEEDED
            if head_preflight and succeeded and not self._cancel.is_set():
                with self._lock:
                    self._last_completed_head_target = (
                        tuple(float(value) for value in positions),
                        int(self.get_clock().now().nanoseconds),
                    )
            return succeeded
        finally:
            if not keep_goal_registered:
                with self._lock:
                    if goal_handle in self._goal_handles:
                        self._goal_handles.remove(goal_handle)
