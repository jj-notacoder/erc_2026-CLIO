# Verify the official simulator source

Current reference: organizer repository `dfl-rlab/erc_sim_2026`, commit
`b1f9b05e20f4750b59f3321d88f172cc4dbb1386`. The [reference decision](compliance_reference.md)
links the published authority and explains why current main is newer than the
latest numbered release. [official_environment.json](../../official_environment.json)
records the adopted asset hashes.

The included checker is read-only with respect to the source tree. It starts
no ROS nodes or simulator processes, publishes no commands, and never repairs
or replaces a file. It returns exit status 0 on a match and 1 on a mismatch.

From the repository root:

```bash
python3 tools/official_reference/check_official_integrity.py check \
  --manifest tools/official_reference/official_b1f9b05_source_manifest.json \
  --root . \
  --allow-extra-package erc_phase1_solution \
  --allow-bytecode-header-refresh \
  --output results/official_integrity_before_run.json
```

The default check covers all **887 official files under `src/`**. It allows
the named solution package, rejects changes/missing files and unexpected
additional source files. Untracked output in `__pycache__` and `.pytest_cache`
directories is ignored, but the two officially tracked Python cache files
are still checked. An official package cannot be exempted. It reports and
accepts CRLF-only differences in UTF-8 text by default.

Normal Python 3.10 imports may refresh the 16-byte headers of those two tracked
`.pyc` files. The explicit `--allow-bytecode-header-refresh` option accepts
this only when the Python magic bytes, compiled-payload length and SHA-256
payload hash match the official manifest. Every accepted refresh records the
old and new headers. Payload changes, different magic bytes and missing
payload metadata still fail. Without this option, header changes also fail.

Use `--strict-bytes` on a canonical LF Linux checkout to require exact Git
blob bytes, including text line endings, apart from explicitly enabled and
logged bytecode-header refreshes. Other binary files always require exact
bytes. A Git archive or Linux checkout with
`core.autocrlf=false` preserves those bytes. Add `--include-docker` to check
the eight official Docker files as well. This repository's local rendering
and evidence-mount helpers can differ from upstream, so Docker differences
must be reviewed separately rather than described as an exact match.

Run the source check before and after building and retain both outputs with
the code version and run manifest. Use the organizer's checked-in URDF; do
not regenerate it with the older generator. A matching source tree does not
prove which installed binaries or image were executed, which parameters were
overridden, or whether grasping/delivery succeeded.

The manifest was generated directly from the official Git objects, not from
the team working tree. To reproduce it from a clean official clone:

```bash
python3 tools/official_reference/check_official_integrity.py build-manifest \
  --git-repo /path/to/official/erc_sim_2026 \
  --ref b1f9b05e20f4750b59f3321d88f172cc4dbb1386 \
  --output /tmp/official_b1f9b05_source_manifest.json
```

Historical v1.0.3 runs, cached grasp/route certificates and test counts do not
validate this updated model. No new-model mission success is claimed by
packaging the checker or adopting the official files.

Run the focused checker tests without ROS:

```bash
python3 tools/official_reference/test_check_official_integrity.py
```
