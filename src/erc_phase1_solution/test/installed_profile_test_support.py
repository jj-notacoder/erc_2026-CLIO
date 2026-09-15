"""Strict profile-only inverse for historical config assertions."""
import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent/'fixtures/installed_profile_inverse.json'
FIXTURE_SHA = '78c65f0de18ab688011d81f7650a93bc7f9b7819d4968ae518e0a2550403b9a1'


def selected_profile_record():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA
    return json.loads(raw)


def restore_installed_profile_bytes(source):
    record = selected_profile_record()
    assert hashlib.sha256(source).hexdigest() == record['candidate_sha256']
    for change in reversed(record['replacements']):
        new, old = change['new'].encode(), change['old'].encode()
        assert source.count(new) == change['count']
        source = source.replace(new, old, change['count'])
    assert hashlib.sha256(source).hexdigest() == record['parent_sha256']
    return source
