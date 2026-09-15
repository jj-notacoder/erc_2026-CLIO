# Five-minute timing progress â€” 14 September 2026

The same solution version completed the tested red-book pickup and bin-delivery task twice in under five minutes, including simulator startup. Both completed trials passed the unchanged final-containment check.

| Trial | Solution launch to completion | Simulator start marker to completion | Final containment |
| --- | ---: | ---: | --- |
| Run97 | 4:36.906 | approximately 4:52.546 | Passed; 22 stable samples over 23.006 simulated seconds |
| Run99 | 4:43.676 | approximately 4:58.012 | Passed; 22 stable samples over 20.992 simulated seconds |

These are headless runs on the development computer, using the same task and official simulation assets. The startup marker has one-second precision; build time and the extra post-completion observation period are excluded. A complete five-minute competition video, GUI timing, other target books and a wider reliability study remain unverified.

Run98, between these trials, failed the sensor-readiness check before mission dispatch. Both laser streams and the depth camera image/info were missing. The failed attempt was preserved and its world closed before the fresh Run99 startup. It is not counted as a completed mission or hidden inside the successful timing results.

## Changes

- Shorter initial arm positioning and checked pickup/placement segments.
- Faster convergence during normal navigation, with existing speed caps, stopping checks and obstacle checks retained.
- Less repeated head positioning and geometry/contact-processing work.
- Mission completion at the checked stationary release with positive target/bin contact, followed by independent containment validation. A final arm return is not included.

Official physics, models and controllers are unchanged. These changes were tested together; the time saving is not attributed independently to every item.

## Placement quality and validation

The book was fully inside in both completed trials. Minimum final side clearance was 66.213 mm in Run97 and 0.780 mm in Run99, so landing position still varies. Placement remains a drop: the last measured book bottom before opening was approximately 267 mm and 262 mm above the bin floor. Gentle placement and continuous collision freedom are not established.

The current navigation change passed 534 affected tests and three constructor checks. Earlier affected-module and full-suite evidence is retained separately; a new full-suite run is not claimed. The exact package was built and executed for both completed trials.

## Reproduction and publication

The tested configuration is selected by the normal `solution.launch.py` launch and its default `collision_quality` profile. After building and sourcing the updated package, the tested task uses `shelf_column_number:=2 book_colour:=red`; supply the real team name. The measurement harness used official headless simulation, seed 101 and ROS domain 31.

This release contains the exact 453-file candidate69 / revision117 package used by Run97 and Run99, with the tested configuration selected by the normal public launch. The [compact evidence bundle](evidence/five_minute/README.md) records the two completed results, the intervening Run98 readiness failure and the separate software-validation scope. Historical Run76 evidence remains available in its own bundle.
