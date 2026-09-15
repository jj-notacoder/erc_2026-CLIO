"""Original R45 head method; positive control only, never imported by production.
Node SHA5f3d5ce232dcc79fb6e1713bf54ad8bb489ea34708c7a2f1b420fbf3199dffef.
"""

def _move_head(self, pan: float, tilt: float) -> bool:
    if (
        self._held_book_corners is not None
        and not self._carried_head_transition_is_safe(pan, tilt)
    ):
        raise RuntimeError('Carried book or robot blocks requested head motion')
    return self._follow(
        self.head_client,
        HEAD_JOINTS,
        [pan, tilt],
        1.2,
    )

