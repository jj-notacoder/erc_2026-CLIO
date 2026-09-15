# Historical source fixtures

These two pinned files are read as text/AST by current regression tests. They are never installed as robot runtime code. `full37` contains candidate20's manipulation node; `full41` contains candidate25's completed-torso helper. The Full48 validation used these same file hashes. The test launcher verifies them and supplies `HEAD_RETURN_BASE_SOURCE` and `TORSO_RETRY_BASE_SOURCE`.
