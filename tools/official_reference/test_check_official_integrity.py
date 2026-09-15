"""Focused read-only checker contracts; all edits are isolated fixture copies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import check_official_integrity as checker


class TrackedBytecodeIntegrityTests(unittest.TestCase):
    def setUp(self):
        tool_dir = Path(__file__).resolve().parent
        repo = tool_dir.parents[1]
        full = json.loads((tool_dir / 'official_b1f9b05_source_manifest.json').read_text())
        entry = next(e for e in full['entries']
                     if e['path'].endswith('/__init__.cpython-310.pyc'))
        self.manifest = {**full, 'entries': [entry]}
        self.original = (repo / entry['path']).read_bytes()
        # Tests restore only the fixture's header to its official bytes. The
        # real tracked source cache is read, never rewritten or imported.
        self.original = bytes.fromhex(entry['python_bytecode']['official_header_hex']) + self.original[16:]
        self.temp = tempfile.TemporaryDirectory(prefix='erc_integrity_fixture_')
        self.root = Path(self.temp.name).resolve()
        # Check the resolved cleanup scope before registering recursive cleanup.
        if self.root.parent != Path(tempfile.gettempdir()).resolve():
            raise AssertionError('Fixture directory escaped the intended temporary root')
        self.addCleanup(self.temp.cleanup)
        self.path = self.root / entry['path']
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.original)

    def result(self, enabled=True, manifest=None):
        return checker.check(self.root, manifest or self.manifest, strict_bytes=True,
                             allow_bytecode_header_refresh=enabled)

    def test_original_bytes_match_without_exception(self):
        result = self.result(False)
        self.assertTrue(result['pass'])
        self.assertEqual(result['bytecode_header_refreshes'], [])

    def test_header_refresh_requires_option_and_is_logged(self):
        data = bytearray(self.original)
        data[8] ^= 1
        self.path.write_bytes(data)
        self.assertFalse(self.result(False)['pass'])
        result = self.result()
        self.assertTrue(result['pass'])
        self.assertEqual(len(result['bytecode_header_refreshes']), 1)
        self.assertEqual(result['bytecode_header_refreshes'][0]['observed_header_hex'], data[:16].hex())

    def test_payload_tampering_is_rejected_even_with_header_option(self):
        data = bytearray(self.original)
        data[8] ^= 1
        data[-1] ^= 1
        self.path.write_bytes(data)
        result = self.result()
        self.assertFalse(result['pass'])
        self.assertEqual(len(result['changed']), 1)
        self.assertEqual(result['bytecode_header_refreshes'], [])

    def test_changed_python_magic_is_rejected(self):
        data = bytearray(self.original)
        data[0] ^= 1
        self.path.write_bytes(data)
        self.assertFalse(self.result()['pass'])

    def test_header_refresh_without_pinned_payload_metadata_is_rejected(self):
        data = bytearray(self.original)
        data[8] ^= 1
        self.path.write_bytes(data)
        manifest = copy.deepcopy(self.manifest)
        del manifest['entries'][0]['python_bytecode']
        self.assertFalse(self.result(manifest=manifest)['pass'])


if __name__ == '__main__':
    unittest.main()
