"""Restore the two declared changes for the older structural assertion only."""
import hashlib
import json
from pathlib import Path
from candidate_composition_support import restore_current_extensions

FIXTURES=Path(__file__).resolve().parent/'fixtures'

def restore_empty_pickup_snapshot(source):
    source=restore_current_extensions(source, filename="empty_pickup_collision.py")
    record=json.loads((FIXTURES/'empty_pickup_snapshot_inverse.json').read_text())
    digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
    assert digest(source)==record['candidate']
    for item in reversed(record['replacements']):
        assert source.count(item['after'])==1
        source=source.replace(item['after'],item['before'],1)
    assert digest(source)==record['parent']
    assert source==(FIXTURES/'empty_pickup_snapshot_parent.py').read_text()
    return source
