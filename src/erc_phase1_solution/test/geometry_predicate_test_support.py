"""Strict outer inverse to the complete profiled340 modules."""
import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent/'fixtures/geometry_predicate_inverse.json'
FIXTURE_SHA256 = 'a1c5ade4d76cd096abdbb6df090d521e40ad2b9286f363766277cda870943180'


def restore_predicate_source(source, filename):
    from candidate_composition_support import restore_current_extensions
    source = restore_current_extensions(source, filename)
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FIXTURE_SHA256
    records = json.loads(data)
    name = 'erc_phase1_solution/' + filename
    if name not in records:
        return source  # The immediately following original bridge pins these.
    record = records[name]
    normalize = lambda text: text.replace('\r\n', '\n').replace('\r', '\n')
    source = normalize(source)
    assert hashlib.sha256(source.encode()).hexdigest() == record['candidate_text_sha256']
    for item in reversed(record['replacements']):
        old, new = normalize(item['old']), normalize(item['new'])
        assert source.count(new) == 1
        source = source.replace(new, old, 1)
    assert hashlib.sha256(source.encode()).hexdigest() == record['parent_text_sha256']
    return source


def restore_predicate_bytes(data, filename):
    from candidate_composition_support import restore_current_extensions
    data = restore_current_extensions(data, filename)
    records = json.loads(FIXTURE.read_bytes())
    name = 'erc_phase1_solution/' + filename
    if name not in records:
        return data
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    record = records[name]
    assert hashlib.sha256(data).hexdigest() == record['candidate_sha256']
    source = data.decode()
    for item in reversed(record['replacements']):
        assert source.count(item['new']) == 1
        source = source.replace(item['new'], item['old'], 1)
    assert hashlib.sha256(source.encode()).hexdigest() == record['parent_sha256']
    return source.encode()
