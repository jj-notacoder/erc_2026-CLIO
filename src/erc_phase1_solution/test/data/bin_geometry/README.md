# Bin geometry captures

These synchronized RGB, depth, camera intrinsics and camera-to-base TF captures
come from CLIO ERC revision b9104920f486c7dbb54b2e8f726f2fcee04d3365; see the
package's THIRD_PARTY_NOTICES.md.

`arena_logo_rgbd.npz` is a negative capture after CLIO's failed green/marker-4
return on 2026-09-10: the large red arena logo is not the collection bin.
`real_bin_rgbd.npz` is a positive stationary capture from a fresh simulation
on the same date, with only the head turned toward the container.

The fixtures contain sensor observations and calibration only. They do not
prove a completed mission or randomized perception accuracy.
