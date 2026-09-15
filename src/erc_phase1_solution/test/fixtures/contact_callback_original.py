"""Original final37 contact callback; parse as AST, do not import.
Full original node SHA256: f795acd78a4fd53e62c56486f4757c2b18ca70c64f5f2689f644e2df6a1321fd
Only this exact method is required for differential tests.
"""
class ManipulationNode:
    def _on_contacts(self, message: Contacts) -> None:
        _observe_place_contacts(self, message)
        now_ns = _message_stamp_ns(
            message,
            int(self.get_clock().now().nanoseconds),
            # Contacts can precede /clock delivery. Collapsing distinct physics
            # stamps would turn sequential forces into a false simultaneous sum.
            clamp_small_future=False,
        )
        lock = getattr(self, '_lock', None)
        if lock is None:
            target_model = getattr(self, '_target_book_model', None)
            contact_epoch = int(getattr(self, '_contact_epoch', 0))
        else:
            with lock:
                target_model = getattr(self, '_target_book_model', None)
                contact_epoch = int(getattr(self, '_contact_epoch', 0))
        robot_contact_models = set()
        external_hand_contact = False
        side_contacts: Dict[str, List[bool]] = {}
        side_forces: Dict[str, List[Dict[Tuple[str, str], float]]] = {}
        held_sensor_fault = None
        stock_models = set()
        for contact in getattr(message, 'contacts', []):
            first = getattr(getattr(contact, 'collision1', None), 'name', '')
            second = getattr(getattr(contact, 'collision2', None), 'name', '')
            pair = normalize_contact_pair(first, second)
            # Ignore contacts internal to the finger linkage. Any other
            # left-gripper contact prevents an empty-hand staging certificate.
            if ('gripper_left_' in first.lower()) != ('gripper_left_' in second.lower()):
                external_hand_contact = True
                # Target-only histories omit unrelated pairs. Preserve negative
                # raw observations without relabeling them as target force.
                raw_expected = target_model
                if _stock_diagnostic(self):
                    import re
                    other = second if 'gripper_left_' in first.lower() else first
                    models = [part for part in other.split('::') if re.fullmatch(
                        r'book_col_\d+_row_\d+_(red|green|yellow|blue)', part)]
                    if len(models) == 1 and models[0].endswith('_' + self.target_colour):
                        stock_models.add(models[0])
                        if raw_expected is None: raw_expected = models[0]
                raw_fault = held_contact_identity_fault(first, second, raw_expected)
                raw_force = _contact_force_magnitude(
                    contact, gripper_is_collision1='gripper_left_' in first.lower())
                if raw_force is None:
                    raw_fault = raw_fault or 'invalid_held_contact_force'
                elif not _stock_diagnostic(self) and raw_force > float(getattr(self, 'adaptive_contact_force_maximum', math.inf)):
                    raw_fault = raw_fault or 'force_overload'
                held_sensor_fault = held_sensor_fault or raw_fault
            if is_target_book_non_gripper_robot_contact(
                pair,
                self.target_colour,
                None,
            ):
                observed_model = (
                    book_model_name(first) or book_model_name(second)
                )
                robot_contact_models.add(observed_model or '')
            if not is_intentional_gripper_book_contact(
                pair,
                self.target_colour,
                target_model,
            ):
                continue
            # The mission deliberately parks the right arm.  Contacts from its
            # gripper must never satisfy the left-arm pinch gate merely because
            # both grippers use the same fingertip-side naming convention.
            left_gripper_names = [
                name for name in pair if 'gripper_left_' in name.lower()
            ]
            if not left_gripper_names:
                continue
            observed_model = book_model_name(first) or book_model_name(second)
            model_key = observed_model or target_model or ''
            sides = side_contacts.setdefault(model_key, [False, False])
            forces = side_forces.setdefault(model_key, [{}, {}])
            force_magnitude = _contact_force_magnitude(
                contact,
                gripper_is_collision1=(
                    'gripper_left_' in first.lower()
                ),
            )
            for name in left_gripper_names:
                lowered = name.lower()
                if any(
                    token in lowered
                    for token in (
                        'base_finger_left',
                        'inner_finger_left',
                        'outer_finger_left',
                        'fingertip_left',
                    )
                ):
                    sides[0] = True
                    if force_magnitude is not None:
                        forces[0][pair] = max(
                            forces[0].get(pair, 0.0), force_magnitude,
                        )
                if any(
                    token in lowered
                    for token in (
                        'base_finger_right',
                        'inner_finger_right',
                        'outer_finger_right',
                        'fingertip_right',
                    )
                ):
                    sides[1] = True
                    if force_magnitude is not None:
                        forces[1][pair] = max(
                            forces[1].get(pair, 0.0), force_magnitude,
                        )

        newly_latched_model: Optional[str] = None

        def apply_updates() -> None:
            nonlocal newly_latched_model
            self._empty_hand_contact = (now_ns, external_hand_contact)
            if external_hand_contact:
                self._last_external_hand_contact_ns = now_ns
            if (external_hand_contact
                    and (getattr(self, '_empty_arm_motion_active', False)
                         or getattr(self, '_empty_arm_contact_guard', False))):
                new_empty_contact = not getattr(self, '_empty_arm_contact_latched', False)
                self._empty_arm_contact_latched = True
                # The action polling loop and gripper wait both observe this
                # interlock, even if the contact disappears on the next frame.
                self._cancel.set()
                if new_empty_contact:
                    self._publish_status('empty_arm_hazard', reason='external_hand_contact')
            selected_model = getattr(self, '_target_book_model', None)
            if _stock_diagnostic(self) and getattr(self, '_adaptive_close_active', False):
                fault = held_sensor_fault
                if len(stock_models) > 1 or (selected_model is not None and stock_models - {selected_model}):
                    fault = fault or 'unexpected_stock_contact_identity'
                if fault:
                    self._adaptive_overload_latched = self._adaptive_overload_latched or fault
                elif selected_model is None and len(stock_models) == 1:
                    selected_model = next(iter(stock_models))
                    self._target_book_model = selected_model
                    newly_latched_model = selected_model
            payload_under_control = bool(
                getattr(self, '_held_book_corners', None) is not None
                or getattr(self, '_transport_lock_engaged', False)
            )
            if (payload_under_control and selected_model is not None
                    and target_model == selected_model and held_sensor_fault):
                self._held_grip_sensor_fault = (
                    getattr(self, '_held_grip_sensor_fault', None) or held_sensor_fault)
            matched_held_robot_contact = bool(
                payload_under_control
                and robot_contact_models
                and (
                    selected_model is None
                    or '' in robot_contact_models
                    or (
                        selected_model in robot_contact_models
                    )
                )
            )
            if matched_held_robot_contact:
                # A fresh-retention probe changes the gripper-sample epoch, but
                # it must not erase collision evidence for a payload that is
                # still under control.  A successful release clears the held
                # state and transport lock under this same mutex, so callbacks
                # applied after release cannot re-latch an old collision.
                self._target_robot_contact_latched = True
            # Contact callbacks parse outside the lock.  A successful release
            # or a fresh-retention probe can reset the sample epoch meanwhile;
            # never let a pre-reset message repopulate the cleared state.
            if int(getattr(self, '_contact_epoch', 0)) != contact_epoch:
                if matched_held_robot_contact:
                    self._contact_generation = int(
                        getattr(self, '_contact_generation', 0)
                    ) + 1
                return
            samples = getattr(self, '_book_contact_samples', {})
            for model_key, (saw_left, saw_right) in side_contacts.items():
                timestamps = samples.setdefault(model_key, [0, 0])
                if saw_left:
                    timestamps[0] = max(timestamps[0], now_ns)
                if saw_right:
                    timestamps[1] = max(timestamps[1], now_ns)
            self._book_contact_samples = samples
            force_samples = getattr(
                self,
                '_book_contact_force_samples',
                {},
            )
            force_frames = getattr(self, '_book_contact_force_frames', {})
            for model_key, (left_forces, right_forces) in side_forces.items():
                if not left_forces and not right_forces:
                    continue
                histories = force_samples.get(model_key)
                if histories is None:
                    histories = [deque(maxlen=64), deque(maxlen=64)]
                    force_samples[model_key] = histories
                frame_sides = force_frames.setdefault(model_key, [{}, {}])

                def append_force(
                    side_index: int, values: Dict[Tuple[str, str], float],
                ) -> None:
                    if not values:
                        return
                    history = histories[side_index]
                    frames = frame_sides[side_index]
                    frame = frames.setdefault(now_ns, {})
                    # Gazebo reports all contact points for one collision pair
                    # together. Repeated reports of that same pair/stamp are
                    # duplicates, while different finger parts add real loads.
                    # Keep the larger duplicate to preserve the overload guard.
                    for pair_key, pair_force in values.items():
                        frame[pair_key] = max(frame.get(pair_key, 0.0), pair_force)
                    force = float(sum(frame.values()))
                    if _stock_diagnostic(self) and not math.isfinite(force):
                        if getattr(self, '_adaptive_close_active', False):
                            self._adaptive_overload_latched = self._adaptive_overload_latched or 'invalid_stock_contact_force'
                        if payload_under_control:
                            self._held_grip_sensor_fault = getattr(self, '_held_grip_sensor_fault', None) or 'invalid_stock_contact_force'
                    self._record_adaptive_force_peak(
                        model_key, side_index, force, now_ns,
                    )
                    if (not _stock_diagnostic(self) and payload_under_control and model_key == selected_model
                            and force > float(getattr(
                                self, 'adaptive_contact_force_maximum', math.inf))):
                        # Preserve the existing per-side cap across contact
                        # probes after acquisition, including distinct parts
                        # whose individual forces are each below that cap.
                        self._held_grip_sensor_fault = (
                            getattr(self, '_held_grip_sensor_fault', None) or 'force_overload')
                    if (
                        bool(getattr(self, '_adaptive_close_active', False))
                        and not _stock_diagnostic(self)
                        and force
                        > float(
                            getattr(
                                self,
                                'adaptive_contact_force_maximum',
                                math.inf,
                            )
                        )
                        and getattr(
                            self,
                            '_adaptive_overload_latched',
                            None,
                        )
                        is None
                    ):
                        self._adaptive_overload_latched = 'force_overload'
                    # Different contact publishers can arrive out of order.
                    # A late frame must not erase newer measurements. Explicit
                    # contact-epoch resets clear both stores together.
                    stamps = sorted(frames)
                    for old_stamp in stamps[:-64]:
                        del frames[old_stamp]
                    history.clear()
                    history.extend(
                        ForceSample(stamp, float(sum(frames[stamp].values())))
                        for stamp in stamps[-64:]
                    )

                if left_forces:
                    append_force(0, left_forces)
                if right_forces:
                    append_force(1, right_forces)
            self._book_contact_force_samples = force_samples
            self._book_contact_force_frames = force_frames

            selected_model = getattr(self, '_target_book_model', None)
            if selected_model is None:
                maximum_age_ns = int(
                    float(getattr(self, 'grasp_contact_max_age', 0.75)) * 1e9
                )
                candidates = [
                    (min(timestamps), model)
                    for model, timestamps in samples.items()
                    if model
                    and all(
                        timestamp > 0
                        and -100_000_000 <= now_ns - timestamp <= maximum_age_ns
                        for timestamp in timestamps
                    )
                ]
                if candidates:
                    _, selected_model = max(candidates)
                    self._target_book_model = selected_model
                    newly_latched_model = selected_model

            if selected_model is not None:
                selected = samples.get(selected_model, samples.get('', [0, 0]))
                self._left_target_contact_ns = selected[0]
                self._right_target_contact_ns = selected[1]
            elif '' in samples:
                self._left_target_contact_ns = samples[''][0]
                self._right_target_contact_ns = samples[''][1]

            anonymous_contact_is_held = bool(
                selected_model is None
                and '' in samples
                and all(
                    timestamp > 0
                    and -100_000_000 <= now_ns - timestamp
                    <= int(
                        float(getattr(self, 'grasp_contact_max_age', 0.75))
                        * 1e9
                    )
                    for timestamp in samples['']
                )
            )
            matched_robot_contact = bool(
                robot_contact_models
                and (
                    (
                        selected_model is not None
                        and (
                            selected_model in robot_contact_models
                            or '' in robot_contact_models
                        )
                    )
                    or anonymous_contact_is_held
                )
            )
            if matched_robot_contact:
                # Keep the collision recorded even if the contact bridge stops
                # publishing it before the trajectory watchdog next polls.
                self._target_robot_contact_latched = True

            if matched_robot_contact or side_contacts:
                self._contact_generation = int(
                    getattr(self, '_contact_generation', 0)
                ) + 1

            # Publish while the same lock still protects the identity.  A
            # successful open therefore cannot reset the identity before its
            # latch event is emitted.
            if newly_latched_model is not None:
                publisher = getattr(self, '_publish_status', None)
                if callable(publisher):
                    publisher(
                        'target_book_latched',
                        model=newly_latched_model,
                    )

        if lock is None:
            apply_updates()
        else:
            with lock:
                apply_updates()

        self._interrupt_adaptive_gripper_if_fault()
        controller = getattr(self, '_fine_gripper_controller', None)
        if controller is not None:
            try:
                controller.observe_contacts(message)
            except Exception as exc:
                controller._hold('fine_contact_callback_failed')
                self._publish_status('fine_gripper_progress',
                                     stage='contact_callback_error', detail=str(exc))

