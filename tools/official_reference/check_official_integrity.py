"""Read-only official-source integrity check; never starts ROS or alters assets.

Build a pinned Git-blob manifest from the official repository, then check a source
tree without requiring .git. Default scope is all official src files. A solution
package may be allowed explicitly. Untracked Python/pytest cache output is ignored;
tracked cache files are checked. Optional Python 3.10 bytecode-header refreshes
require unchanged magic bytes and exactly matching compiled payload hashes.
Other added files under src are rejected. CRLF-only text differences are reported
and accepted unless --strict-bytes is supplied. This is source verification, not an
installed-image, runtime-parameter, or competition-score certificate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

OFFICIAL_URL = 'https://github.com/dfl-rlab/erc_sim_2026'
CACHE_DIRS = {'__pycache__', '.pytest_cache'}
PYTHON_310_MAGIC = bytes.fromhex('6f0d0d0a')
BYTECODE_HEADER_SIZE = 16


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def build_manifest(repo, revision):
    origin = git(repo, 'remote', 'get-url', 'origin').decode().strip().removesuffix('.git')
    if origin != OFFICIAL_URL:
        raise ValueError('Manifest source must have the verified official HTTPS origin')
    commit = git(repo, 'rev-parse', revision + '^{commit}').decode().strip()
    entries = []
    for item in git(repo, 'ls-tree', '-r', '-z', commit).split(b'\0'):
        if not item:
            continue
        metadata, path = item.split(b'\t', 1)
        mode, kind, sha = metadata.decode().split()
        path = path.decode('utf-8')
        if path.startswith(('src/', 'docker/')):
            if kind != 'blob':
                raise ValueError('Unsupported non-blob source entry: ' + path)
            entry = {'path': path, 'mode': mode, 'git_blob_sha1': sha}
            if path.endswith('.pyc'):
                data = git(repo, 'cat-file', 'blob', sha)
                if len(data) >= BYTECODE_HEADER_SIZE and data[:4] == PYTHON_310_MAGIC:
                    entry['python_bytecode'] = {
                        'version': '3.10', 'magic_hex': data[:4].hex(),
                        'header_size': BYTECODE_HEADER_SIZE,
                        'official_header_hex': data[:BYTECODE_HEADER_SIZE].hex(),
                        'payload_size': len(data) - BYTECODE_HEADER_SIZE,
                        'payload_sha256': hashlib.sha256(data[BYTECODE_HEADER_SIZE:]).hexdigest()}
            entries.append(entry)
    return {'schema_version': 1, 'official_repository': OFFICIAL_URL,
            'official_commit': commit, 'git_object_format': 'sha1',
            'scope': 'All tracked src and docker files; check defaults to src only.',
            'entries': entries}


def check(root, manifest, allowed_packages=(), strict_bytes=False, include_docker=False,
          allow_bytecode_header_refresh=False):
    root = Path(root).resolve()
    if manifest.get('official_repository') != OFFICIAL_URL:
        raise ValueError('Unexpected official repository identity')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('official_commit', '')):
        raise ValueError('Manifest must pin a full commit SHA')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', p) for p in allowed_packages):
        raise ValueError('Allowed packages must be single safe directory names')
    prefixes = ('src/', 'docker/') if include_docker else ('src/',)
    expected = {e['path']: e for e in manifest['entries'] if e['path'].startswith(prefixes)}
    official_packages = {p.split('/')[1] for p in expected if p.startswith('src/')}
    if official_packages.intersection(allowed_packages):
        raise ValueError('Cannot exempt an official package')
    missing, changed, eol_only, extra, header_refreshes = [], [], [], [], []
    exact = 0
    for relative, entry in expected.items():
        path = root.joinpath(*relative.split('/'))
        if not path.exists() and not path.is_symlink():
            missing.append(relative)
            continue
        if entry['mode'] == '120000':
            data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        elif path.is_symlink():
            changed.append({'path': relative, 'reason': 'unexpected symlink'})
            continue
        else:
            data = path.read_bytes()
        actual = blob_hash(data)
        if actual == entry['git_blob_sha1']:
            exact += 1
            continue
        bytecode = entry.get('python_bytecode', {})
        if (allow_bytecode_header_refresh and relative.endswith('.pyc')
                and bytecode.get('version') == '3.10'
                and bytecode.get('header_size') == BYTECODE_HEADER_SIZE
                and data[:4] == PYTHON_310_MAGIC
                and data[:4].hex() == bytecode.get('magic_hex')
                and len(data) == BYTECODE_HEADER_SIZE + bytecode.get('payload_size', -1)
                and hashlib.sha256(data[BYTECODE_HEADER_SIZE:]).hexdigest()
                == bytecode.get('payload_sha256')):
            header_refreshes.append({
                'path': relative,
                'official_header_hex': bytecode['official_header_hex'],
                'observed_header_hex': data[:BYTECODE_HEADER_SIZE].hex(),
                'payload_sha256': bytecode['payload_sha256']})
            continue
        text = False
        if not strict_bytes and b'\0' not in data:
            try:
                data.decode('utf-8')
                text = True
            except UnicodeDecodeError:
                pass
        if text and blob_hash(data.replace(b'\r\n', b'\n')) == entry['git_blob_sha1']:
            eol_only.append(relative)
        else:
            changed.append({'path': relative, 'expected_git_blob_sha1': entry['git_blob_sha1'],
                            'actual_git_blob_sha1': actual})
    for scope in prefixes:
        base = root / scope.rstrip('/')
        if not base.is_dir():
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in CACHE_DIRS]
            rel_dir = Path(directory).relative_to(root)
            if len(rel_dir.parts) == 1 and rel_dir.parts[0] == 'src':
                dirs[:] = [d for d in dirs if d not in allowed_packages]
            for name in names:
                relative = (Path(directory) / name).relative_to(root).as_posix()
                if relative not in expected:
                    extra.append(relative)
    return {'official_repository': OFFICIAL_URL, 'official_commit': manifest['official_commit'],
            'checked_source_root': str(root), 'scope': list(prefixes),
            'allowed_extra_solution_packages': sorted(allowed_packages),
            'ignored_cache_directories': sorted(CACHE_DIRS), 'strict_bytes': strict_bytes,
            'allow_bytecode_header_refresh': allow_bytecode_header_refresh,
            'bytecode_header_refreshes': header_refreshes,
            'expected_files': len(expected), 'exact_matches': exact,
            'crlf_only_matches': sorted(eol_only), 'missing': sorted(missing),
            'changed': changed, 'unexpected_files': sorted(extra),
            'pass': not missing and not changed and not extra,
            'limits': 'Checks source files only, not installed binaries, image digest, runtime environment, overridden parameters, or scoring.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build-manifest')
    build.add_argument('--git-repo', required=True, type=Path)
    build.add_argument('--ref', required=True)
    build.add_argument('--output', required=True, type=Path)
    verify = sub.add_parser('check')
    verify.add_argument('--manifest', required=True, type=Path)
    verify.add_argument('--root', required=True, type=Path)
    verify.add_argument('--allow-extra-package', action='append', default=[])
    verify.add_argument('--strict-bytes', action='store_true')
    verify.add_argument('--include-docker', action='store_true')
    verify.add_argument('--allow-bytecode-header-refresh', action='store_true',
                        help='Allow tracked Python 3.10 pyc headers to refresh only '
                             'when magic and compiled payload match the official manifest')
    verify.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'build-manifest':
        result = build_manifest(args.git_repo, args.ref)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'manifest': str(args.output), 'commit': result['official_commit'],
                          'entries': len(result['entries'])}))
        return 0
    result = check(args.root, json.loads(args.manifest.read_text(encoding='utf-8')),
                   args.allow_extra_package, args.strict_bytes, args.include_docker,
                   args.allow_bytecode_header_refresh)
    rendered = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
