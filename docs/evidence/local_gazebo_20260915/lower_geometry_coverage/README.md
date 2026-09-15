# Lower-row geometry coverage

These controller-free probes used production source `0a5fa41`. They do not
establish successful physical delivery or live timing.

| Scope | Result | Offline wall time |
|---|---|---:|
| Synthetic printed 3/yellow tight-lateral PICK, extraction and carry | Passed | 32.980 s |
| Synthetic printed 3/green bottom tight-lateral PICK, extraction and carry | Passed | 57.841 s |
| Row-2 PLACE using saved carry plus recorded blue bin/table context | Passed full preflight | 53.457 s |
| Bottom PLACE using saved 3/green carry plus recorded blue bin/table context | Failed at 512-solve IK limit | 16.203 s |

`lateral_pick/` records complete guarded pickup, carry and look-bin geometry,
with synthetic onboard-aligned marker/book relationships. `lower_place/` records
mixed-fixture PLACE inputs and full candidate checks. Its README distinguishes
older row-2 carry evidence and the corrected initial fixture-phase assumption.
Both PLACE fixtures exactly reconstructed their saved attached corners using the
official book dimensions. Production predicates and search limits were unchanged.

Each source harness retains its original temporary-path invocation. To replay,
restore its documented `/tmp` inputs in the prepared container. Source identities,
fixture assumptions, controller-call counts and owner cleanup are recorded in
individual JSON files. No physical trial was running during these probes.
