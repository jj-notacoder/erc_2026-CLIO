"""Exact simulation-clock inverse for historical preopen source comparisons.

Behavioral fixtures continue to execute the current production helper, including
its bounded no-progress watchdog while simulation time advances.
"""
from candidate_composition_support import (
    restore_preopen_timing_source as _restore_preopen_timing_source,
)


def restore_preopen_timing_source(source):
    replacements = (
        (
            '    Ten wall seconds maximum, including a fixed-frame clock catch-up with a\n'
            '    0.5 wall-second no-progress watchdog when simulation time is active;\n',
            '    Ten wall seconds maximum, including a <=0.5 s fixed-frame clock catch-up;\n',
        ),
        (
            "    simulated = bool(getattr(node.get_clock(), 'ros_time_is_active', False))\n",
            '',
        ),
        (
            '                if simulated and updated > now:\n'
            '                    # A permitted 100 ms ROS lead takes more than 0.5 wall\n'
            '                    # seconds below RTF 0.2. Keep waiting for this fixed frame\n'
            '                    # while /clock advances, within the original total bound.\n'
            '                    catchup = min(deadline, time.monotonic()+.5)\n',
            '',
        ),
    )
    for current, historical in replacements:
        assert source.count(current) == 1, current
        source = source.replace(current, historical, 1)
    # The existing inverse still requires the complete original candidate and
    # parent hashes; extra changes cannot disappear through this composition.
    return _restore_preopen_timing_source(source)
