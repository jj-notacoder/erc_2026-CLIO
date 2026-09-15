# Revision46 timing improvements

Revision46 was not promoted: full validation finished with 3,003 passing tests and one obsolete fixture-scope assertion. The failed assertion reversed the body-separation change and compared the entire node against R43, unintentionally rejecting the new head, torso and executor changes. Both selected/default four-node constructor checks, official pre/post checks, all 40 mesh checks and source preservation passed. The failed snapshot is retained; a successor must pass fresh validation. No physical delivery on these bytes has been demonstrated. The latest completed physical delivery remains Run40 on revision42, at 42 min 10.733 s. Run41 on revision43 was cancelled during stationary placement planning after missing the target.

The 290-file package is frozen at tree `1cd96b71fab70549024966130dec6329938dc9ddda233dfcc6fc5e3dceccffd9`. The selected runtime profile remains `590bc458a136cd9d2ab9073104cd496077e5b2df65f0079ed4010a878dedd705`. Official assets, controllers and physics are unchanged.

The package combines four changes:

- Geometry: skip empty filtered SAT work while retaining complete-mesh containment; use sufficient current-vertex separation checks before detailed PICK tool/body and moving-body intersections. Uncertain separation still takes the original detailed path.
- Torso: a fresh, nearly stationary measured height within 1 micrometre of the configured height can become the fixed height of the entire PLACE plan. IK, staging, HOME-at-height, return and the normal 2.2-second torso action use that same coordinate. Freshness, velocity and drift are checked again immediately before sending the action. The nominal configuration is not changed and the torso action is not skipped.
- Head: a repeated retained-book head target can use a fresh measured hold after current body/book/head geometry and contact, cancellation and action-state checks. Otherwise the original full sweep and action remain. The sensor lock is non-reentrant; the candidate's regression tests exercise that actual primitive.
- Executor: manipulation uses a SingleThreadedExecutor. Its command worker remains separate from subscription/action callbacks; controller actions, cancellation and shutdown keep their existing handling.

Focused evidence includes 389 geometry cases, 366 torso/motion/recovery cases, 47 standalone and 47 portable head cases, and 11 executor-main cases. These overlap existing tests and must not be added as a total number of unique tests.

The unchanged-input R45 nominal placement replay passed all 2,337 sample-argument/verdict checks and returned the same paths as the recorded R43 replay. It took 92.517 s wall / 89.976 s thread CPU with cProfile disabled. The R43 reference had cProfile enabled, so the timings are not a matched speed comparison. This replay did not execute R46's torso/head/executor changes or demonstrate physical delivery.

A separate real-DDS executor comparison passed the functional TF, action, cancellation and lost-contact checks. Its finite sensor streams had some best-effort message losses, so strict equal-input admission failed. The observed 96-sample worker timings, 18.280 s for Multi4 and 12.836 s for Single, are preliminary component evidence rather than a matched speedup or full-mission result.

The head hold rejects an incoherent final contact/feedback epoch conservatively. Such rejection can fail a `look_bin` command after a benign sensor update; it is not proof of a physical collision or lost book. The physical trial must distinguish this outcome from an actual hazard. A later fallback, if needed, must retain a fresh complete motion preflight and the real hazard/cancellation stops.

Run42 is prepared to use the ordinary unprofiled team launcher on the same seed 101, column 2/red task. Completion time, actual whole-book containment and support, bin movement, measured empty-hand return, contacts and sampled geometry remain to be assessed independently of a controller success event.
