# Historical v1.0.3 certificate assets

These two files are exact Git blobs from the official simulator's `v1.0.3`
commit `0a09806ecbade5edc9f8a148b7c9f439ed761554`. Their paths and SHA256 values
are recorded in `provenance.json`.

They preserve the 30 mm book and robot definition against which the historical
seed-101 collision certificate was checked. Tests use them to verify that the
unchanged certificate accepts its original assets and rejects either current
`b1f9b05` book/robot replacement. All other certificate inputs are unchanged
between those official revisions and remain bound by the original hashes.

These are test fixtures only. They are not installed as simulator assets and do
not certify the historical route under current physics. Source:
[official v1.0.3](https://github.com/dfl-rlab/erc_sim_2026/tree/0a09806ecbade5edc9f8a148b7c9f439ed761554).
