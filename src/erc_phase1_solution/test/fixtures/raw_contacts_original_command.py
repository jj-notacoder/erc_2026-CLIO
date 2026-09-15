# Exact R52 _on_command extract; full source SHA256 06bef5efb58b73301aced8e942a1df5528a931a3d2ce46e74d307fa29a5810bf
from __future__ import annotations

def _on_command(self, message: String) -> None:
        payload = decode_event(message.data)
        command = str(payload.get('event', message.data)).strip().lower()
        correlation = ({key: payload.get(key) for key in
            ('trial_id', 'placement_attempt_id', 'target_model')}
            if command == 'place' and getattr(self, 'delivery_evidence_enabled', False)
            else {})
        if command in ('cancel', 'abort', 'stop'):
            self._empty_arm_preparation = None
            self._cancel.set()
            with self._lock:
                goal_handles = list(self._goal_handles)
            for goal_handle in goal_handles:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
            if _stock_diagnostic(self) and (getattr(self, '_adaptive_close_active', False)
                    or getattr(self, '_held_book_corners', None) is not None):
                self._hold_adaptive_gripper('cancelled')
            self._publish_status('cancelled')
            return
        with self._lock:
            if self._busy:
                self._publish_status('rejected', command=command, reason='busy', **correlation)
                return
            if getattr(self, '_pending_retained_acceptances', ()):
                self._publish_status(
                    'rejected', command=command,
                    reason='retained_goal_acceptance_unresolved', **correlation,
                )
                return
            if self._goal_handles:
                # A handle remains registered while a controller goal is
                # active, including when cancellation could not be confirmed.
                # Do not clear the cancel interlock and issue a conflicting
                # command from that unknown physical state.
                self._publish_status(
                    'rejected',
                    command=command,
                    reason='controller_goal_unresolved', **correlation,
                )
                return
            self._busy = True
            self._cancel.clear()
        _cpu_command_target = self._run_command
        try:
            _cpu_command_target = _planning_cpu_command_target(self, _cpu_command_target)
        except Exception:
            pass
        threading.Thread(
            target=_cpu_command_target,
            args=(command, dict(payload)),
            name=f'erc-manipulation-{command}',
            daemon=True,
        ).start()
