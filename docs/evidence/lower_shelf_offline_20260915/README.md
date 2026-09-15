# Lower-shelf offline planning evidence

These jobs issue no controller commands and establish no physical delivery or
mission-time result. They check the unchanged official robot geometry against
registered local shelf bounds and the existing lift/carry predicates.

- `row2_complete.json`: current production helper with the final bounded row-two
  setup family, grasp vertical offset +0.005 m; 56.12 s.
- `row3_complete.json`: integrated candidate with the corrected one-intermediate
  setup, using the recorded column-4/blue observation; 53.32 s. The later shared
  setup adds a row-two fallback after the unchanged successful row-three prefix;
  the later helper also adds a shelf-metrics status field.
- `bottom_complete.json`: registered bottom approach with preserved grasp and
  separate withdrawal; full guard harness, 70.23 s. Source hashes in the file
  identify the draft used. Current-source portable regressions supersede the
  draft checks for integration.

The portable `test_lower_shelf_pick.py` and `test_bottom_registered_official.py`
exercise complete current-source routes. The latter includes an independent
fixed marker fixture, original grasp/withdrawal assertions, and bin-look sweep.
The local shelf bounds explicitly assume an upright supported book. Unknown
neighbouring objects and the full unobserved shelf mesh remain outside that
model; live contact and retention checks remain necessary.
