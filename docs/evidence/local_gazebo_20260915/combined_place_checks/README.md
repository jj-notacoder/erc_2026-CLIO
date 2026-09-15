# Combined placement software checks

**714 tests passed in 15.57 s** across the 24 modules listed in
`source_manifest.json`, against the combined production source. There are no
outstanding failures in this selection. This is not a full-suite run or a
physical delivery claim.

Coverage includes the registered target solver, both wrist policies, bottom-row
proposal selection, complete PLACE flow, torso arrival, arm timing/stationarity,
release-only checks, preserved torso-hold behavior and journal diagnostics.

The first run passed 713 and failed one old test which prohibited every negative-
wrist alternate target. That prohibition was deliberately extended for bottom
placement. The replacement test requires positive-wrist IK results to remain
rejected under negative policy, with the same bounded search and no scene or
candidate admission. The complete selected suite then passed. Both logs are
retained. Source and test hashes bind the final result to the reviewed bytes.
