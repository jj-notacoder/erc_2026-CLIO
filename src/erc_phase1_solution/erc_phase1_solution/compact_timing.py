"""Bounded optional floor for only the supported post-roll compact group."""
import math


def checked_supported_compact_minimum_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not .175 <= seconds <= .35:
        raise ValueError(
            'supported_compact_minimum_segment_seconds must be finite and within [.175, .35]'
        )
    return seconds
