# Registered shelf cache: official geometry A/B

The original and cached evaluators produced **identical results for all 872
ordered samples** on the saved column-4/blue trial `121f7ff7e8ad` route. Every
sample's joint bytes, entry mode, verdict, count and cumulative clearance metric
bytes matched. The final 558-sample clearance metrics also exactly matched the
previously admitted complete plan.

## Measured time

Each mode ran once in a fresh Python process with new official model objects and
empty process caches. OS/filesystem caches were not flushed. Times below are
host wall time inside the registered checks, including the same trace observer;
model construction is excluded and recorded separately in the result files.

| Scope | Original | Cached |
| --- | ---: | ---: |
| Saved torso edge and accepted setup | 3.0119 s | 3.2969 s |
| Fresh final setup and Cartesian entry checks | 5.7877 s | 3.7154 s |
| Combined registered checks | **8.7997 s** | **7.0123 s** |

The combined check was 20.31% faster in this pair; the final stage was 35.81%
faster. Fresh identity checks add cost on initial misses. The cached run recorded
259 exact hits, 613 misses and zero unsupported-input fallbacks.

## Inputs and method

The [onboard inputs](../lower_planning_parallel/onboard_inputs.json) and
[common plan](../lower_planning_parallel/common_plan.json) are shared with the
earlier lower-planning comparisons. Their parsed values were checked against
the original replay inputs exactly; this directory does not duplicate them.

1. Construct the actual `_official_manipulation_planner` fixture and immutable
   meshes from the official robot description. Install the recorded torso,
   left/right arm, head and aperture values. Use fixed replay clock/stamp values;
   this does not validate live freshness.
2. Transform the recorded marker and inward normal using the recorded odometry
   pose. Preserve the 0.200 m lateral uncertainty, 0.010 m roof uncertainty,
   0.010 m normal uncertainty and all existing bounds and allowances.
3. Assert that saved approach solutions and setup waypoints equal the recorded
   plan exactly. Run the unchanged dense registered checks over the torso and
   accepted setup edges.
4. Construct a fresh final bounds accumulator and check the same setup edges
   followed by every Cartesian entry edge. Only the cached mode shares exact
   per-sample results with its prior bounds instance. The sample grid and
   cancellation checks remain active.
5. Compare complete sample traces, model signatures, inputs, route and metrics
   exactly. Verify the same model signature after evaluation. Both processes
   finished within their 120-second limits, with one BLAS/OMP thread and no IK
   or controller calls.

Final minimum clearances were floor **0.0925652624295582 m**, roof
**0.0759881809536056 m**, side **0.18586462024809478 m**, and back
**0.16688242189703822 m**.

## Evidence and scope

- [Comparison](official_comparison.json) contains exact-match results and timings.
- [Original result](serial_official.json) and [cached result](memo_official.json)
  preserve inputs, complete checked route, source/model hashes and phase metrics.
- [Provenance](provenance.json) pins promoted source/test files, the original
  harness, shared inputs, original reports and the matching full sample trace.
  Duplicate traces are omitted. Positive infinite setup minima use the JSON
  string `+Infinity`: no surface entered the protected shelf half-space, so those
  bay-clearance minima were not evaluated. Other report values are unchanged.

The baseline bounds hash is `06e6b8d3…`; the evaluated cached bounds hash is
`d6937f3f…`. Full hashes are in provenance. The draft also passed 138 lightweight
tests after the final scalar-identity correction.

The promoted source subsequently passed 167 selected registered/lower checks,
including complete official-model lower-row candidates, and 128 separate
placement-journal and historical-composition checks. Selection paths overlap;
these are not a new full-suite count or an isolated timing benchmark. The
[production software record](production_software_validation.json) pins the
checked bytes and links both logs. PLACE planning-stage and failure details
are now retained in the mission journal as diagnostic events.

This replay covers registered shelf geometry for one saved route. It does not
rerun setup search, robot self-collision admission, lift/carry planning, sensor
subscriptions or physical motion. Those predicates remain in their existing
callers. It does not establish full planning latency, mission success, collision
freedom during a physical trial, or completion under five minutes. No geometry,
IK, simulation or tests were rerun while assembling this evidence directory.
