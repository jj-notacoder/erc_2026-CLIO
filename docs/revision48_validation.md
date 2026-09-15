# Revision48 timing and head fallback work

Revision48 failed full final13 validation: 3,120 tests passed and 24 failed, with no errors or skips. It was not promoted or run physically. Selected/default four-node construction, 895 official files, 40 mesh checks and unchanged source all passed. The failures are old extracted-method fixture globals and a stale structural assertion; narrow test-only corrections are being prepared for a separate revision and full validation. The failed snapshot remains preserved.

The candidate contains 298 files, source tree `609e9ecaf175aaacf80a76799ee0eda57f086e1c0627b0a595b4752c69d978a6`, with unchanged runtime profile `590bc458a136cd9d2ab9073104cd496077e5b2df65f0079ed4010a878dedd705`. Official robot assets, controllers, physics, shelf withdrawal timing and collision thresholds are unchanged.

The repeated head-look fallback now waits up to two wall seconds for a fresh stationary start when the no-motion shortcut loses eligibility, then performs the ordinary checked head action. Cancellation, changed possession identity, unresolved goals, unsafe geometry and actual latched hazards still stop the command. New reason telemetry distinguishes a fallback from a hard rejection. Run42's exact internal failed branch was not logged, so this source correction is not yet a verified physical fix.

Fixed robot meshes now have explicit immutable local-data handles. A fresh scene sample can also reuse its own world mesh surfaces and exact bounds for the body and scene checks. The original sample sequence, collision predicates and unsupported-input fallbacks remain. The combined source and four portable test/fixture additions were independently compared with their reviewed inputs.

One matched cold benchmark used the same 96 recorded poses on the actual official meshes. Wall time fell from 4.016599 to 3.278248 seconds (18.38%); thread CPU fell from 3.906449 to 3.188109 seconds (18.39%). Pose/aperture/loaded inputs, collision verdicts, first rejection and minimum-height outputs matched exactly. Counters confirmed 23 immutable model records, 4,416 explicit handle arguments, 96 sample captures, and 75 body/scene reuses. The other 21 samples used the existing full-body cache. This is a fixed-context sample comparison, not full-planner or mission timing.

Perception startup selects one OpenCV thread. A separate saved-camera comparison found identical outputs and roughly 31.4% lower CPU use, with essentially unchanged isolated wall time. That comparison does not establish a live Gazebo speedup.

Focused checks passed: 116 head fallback cases, 171 immutable-mesh/local-bound/scene cases, 56 portable sample/body cases and 6 perception-startup/shutdown cases. These earlier passes do not replace the failed full combined run. A passing full validation and a complete timed physical run are still required. The latest full delivery remains Run40 at 42 min 10.733 s. The complete 15-minute target has not been achieved.
