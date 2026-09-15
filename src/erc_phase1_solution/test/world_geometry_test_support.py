"""Strict source bridge to R57, used only by structural/differential tests."""
import hashlib
import json
from pathlib import Path


FIXTURE = Path(__file__).resolve().parent / 'fixtures/world_geometry_inverse.json'
FIXTURE_SHA256 = '35d0c370bc1f740ba6fa292dcd886e712b58ba142ca7fd9f2b5f9524d6b5872a'


def restore_world_geometry_source(source, filename):
    from geometry_predicate_test_support import restore_predicate_source
    source = restore_predicate_source(source, filename)
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FIXTURE_SHA256
    record = json.loads(data)['erc_phase1_solution/' + filename]
    # Text readers may translate only line endings; no semantic normalization.
    normalize = lambda text: text.replace('\r\n', '\n').replace('\r', '\n')
    source = normalize(source)
    assert hashlib.sha256(source.encode()).hexdigest() == record['candidate_text_sha256']
    for item in reversed(record['replacements']):
        old, new = normalize(item['old']), normalize(item['new'])
        assert source.count(new) == 1, new[:100]
        source = source.replace(new, old, 1)
    assert hashlib.sha256(source.encode()).hexdigest() == record['parent_text_sha256']
    return source
