"""Portable identity of installed geometry modules and normal ROS share files.

No ROS import or audit directory is required. The parent resolves package shares
with its normal package index; fresh children validate a JSON descriptor against
the same installed files. This pins file state and import origins, not live scene
freshness or an atomic filesystem snapshot. Call verify before and after loading.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from types import MappingProxyType
import xml.etree.ElementTree as ET


PACKAGE = 'erc_phase1_solution'
SCHEMA = 'installed_geometry_identity_v1'
MAX_MODULES = 512
MAX_ASSETS = 512
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_DESCRIPTION = {
    'urdf': 'urdf/tiago_pro.urdf',
    'bin_mesh': 'models/collection_bin/meshes/erc_base_collection_bin.STL',
    'bin_sdf': 'models/collection_bin/sdf/erc_collection_bin.sdf',
    'table_mesh': 'models/table/meshes/erc_base_table.STL',
    'table_sdf': 'models/table/sdf/erc_table.sdf',
}
# Same stock-model admission as TableSceneObstacle and the original node.
_TABLE_MESH = '9a662650c305d09e67d112b1c3311c78c8b75454eb05c8c7981292eb1d354977'
_TABLE_SDFS = frozenset((
    '90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f',
    '04f1940665775cdff35cdc89ac3518c5db3cbb4337891d9a601b8906c857417f',
))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _absolute(path):
    value = Path(path)
    if not value.is_absolute():
        raise ValueError('absolute installed path required')
    return Path(os.path.abspath(value))


def _file(path):
    logical = _absolute(path)
    resolved = logical.resolve(strict=True)
    first = resolved.stat()
    if not resolved.is_file() or not 0 <= first.st_size <= MAX_FILE_BYTES:
        raise ValueError('missing or oversized installed file: ' + str(logical))
    h = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    last = resolved.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if stamp(first) != stamp(last) or logical.resolve(strict=True) != resolved:
        raise ValueError('installed file changed while hashing: ' + str(logical))
    return dict(path=str(resolved), sha256=h.hexdigest(), size=last.st_size)


def _relative(text):
    if not isinstance(text, str) or not text or '\\' in text or ':' in text:
        raise ValueError('invalid resource path')
    path = PurePosixPath(text)
    if path.is_absolute() or any(x in ('', '.', '..') for x in text.split('/')):
        raise ValueError('resource path escapes package')
    return path


def _mesh_uri(uri, *, sdf=False):
    prefix = 'package://'
    if sdf and isinstance(uri, str) and uri.startswith('model://'):
        prefix = 'model://'
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError('unsupported mesh URI')
    value = uri[len(prefix):]
    # Official SDFs use both model://package and model:///package.
    if sdf and prefix == 'model://':
        value = value.lstrip('/')
    if '/' not in value:
        raise ValueError('mesh URI has no resource')
    package, relative = value.split('/', 1)
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', package):
        raise ValueError('invalid mesh package')
    return package, _relative(relative).as_posix()


def _urdf_meshes(path):
    root = ET.parse(path).getroot()
    if root.tag != 'robot':
        raise ValueError('normal robot URDF required')
    # Include visual resources too: a changed normal robot installation cannot
    # silently select a second model closure, even if this owner uses collision.
    return tuple(_mesh_uri(mesh.get('filename')) for mesh in root.findall('./link/collision/geometry/mesh')
                 + root.findall('./link/visual/geometry/mesh'))


def required_package_names(description_share):
    """Parent may resolve these names with ament; this function imports no ROS."""
    description = _absolute(description_share)
    urdf = description / _DESCRIPTION['urdf']
    _file(urdf)
    names = {'erc_description'}
    names.update(package for package, _ in _urdf_meshes(urdf))
    for key in ('bin_sdf', 'table_sdf'):
        sdf = description / _DESCRIPTION[key]
        _file(sdf)
        names.update(_mesh_uri(mesh.text, sdf=True)[0]
                     for mesh in ET.parse(sdf).getroot().findall('.//geometry/mesh/uri'))
    return tuple(sorted(names))


def _module_inventory(directory):
    directory = _absolute(directory)
    if not directory.is_dir() or not (directory / '__init__.py').is_file():
        raise ValueError('ordinary installed Python package required')
    bindings = {}
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', 'test', 'tests') and not d.startswith('.'))
        # A whole package-directory symlink works; nested package-directory
        # symlinks would otherwise be silently omitted by os.walk.
        if any((Path(root) / d).is_symlink() for d in dirs):
            raise ValueError('nested symlink package directory unsupported')
        for filename in sorted(files):
            if not filename.endswith('.py') or filename.startswith('test_'):
                continue
            relative = (Path(root) / filename).relative_to(directory)
            parts = list(relative.with_suffix('').parts)
            if parts[-1] == '__init__':
                parts.pop()
            if any(not part.isidentifier() for part in parts):
                raise ValueError('invalid installed module name')
            name = '.'.join((PACKAGE, *parts))
            if name in bindings or len(bindings) >= MAX_MODULES:
                raise ValueError('duplicate or excessive installed module inventory')
            bindings[name] = dict(relative=relative.as_posix(), **_file(directory / relative))
    if PACKAGE not in bindings or PACKAGE + '.installed_geometry_identity' not in bindings:
        raise ValueError('installed identity module is absent')
    return bindings


def _capture(directory, shares):
    directory = _absolute(directory)
    if not isinstance(shares, dict) or 'erc_description' not in shares:
        raise ValueError('normal package-share map required')
    names = required_package_names(shares['erc_description'])
    if not set(names).issubset(shares):
        raise ValueError('referenced package share missing')
    packages = {name: str(_absolute(shares[name])) for name in names}
    module_bindings = _module_inventory(directory)
    required = {('erc_description', relative) for relative in _DESCRIPTION.values()}
    required.update((name, 'package.xml') for name in names)
    description = Path(packages['erc_description'])
    required.update(_urdf_meshes(description / _DESCRIPTION['urdf']))
    for key in ('bin_sdf', 'table_sdf'):
        required.update(_mesh_uri(mesh.text, sdf=True) for mesh in
                        ET.parse(description / _DESCRIPTION[key]).getroot().findall('.//geometry/mesh/uri'))
    if len(required) > MAX_ASSETS:
        raise ValueError('excessive geometry resource inventory')
    assets = {}
    for package, relative in sorted(required):
        if package not in packages:
            raise ValueError('referenced package share missing')
        logical = Path(packages[package]) / _relative(relative)
        item = _file(logical)
        assets[package + '/' + relative] = item
    for name in names:
        if ET.parse(Path(packages[name]) / 'package.xml').getroot().findtext('name') != name:
            raise ValueError('resolved package name mismatch')
    total = sum(row['size'] for row in module_bindings.values()) + sum(row['size'] for row in assets.values())
    if total > MAX_TOTAL_BYTES:
        raise ValueError('geometry identity byte budget exceeded')
    source = dict(package=PACKAGE, package_directory=str(directory), modules=module_bindings)
    models = dict(packages=packages, assets=assets, resources=_DESCRIPTION)
    return dict(schema=SCHEMA, **source, packages=packages, assets=assets,
                source_id=_digest(source), model_id=_digest(models))


class InstalledGeometryIdentity:
    """Immutable installed source/resource bindings, serializable without ROS."""
    __slots__ = ('_descriptor', 'source_id', 'model_id', 'package_directory',
                 'packages', 'module_files', 'module_sha256', 'asset_files',
                 'urdf', 'bin_mesh', 'bin_sdf', 'table_mesh', 'table_sdf', '_sealed')

    @classmethod
    def capture(cls, *, package_shares, package_directory=None):
        if package_directory is None:
            package_directory = Path(__file__).parent
        return cls.from_descriptor(_capture(package_directory, dict(package_shares)))

    @classmethod
    def from_descriptor(cls, descriptor):
        if type(descriptor) is not dict or set(descriptor) != {
            'schema', 'package', 'package_directory', 'modules', 'packages',
            'assets', 'source_id', 'model_id',
        } or descriptor.get('schema') != SCHEMA or descriptor.get('package') != PACKAGE:
            raise ValueError('invalid installed identity descriptor')
        # Re-derive complete closure rather than trusting caller-supplied paths,
        # pins, digests, omitted modules, or extra arbitrary resources.
        actual = _capture(descriptor['package_directory'], descriptor['packages'])
        if _canonical(actual) != _canonical(descriptor):
            raise ValueError('installed descriptor does not match current files')
        self = object.__new__(cls)
        object.__setattr__(self, '_descriptor', _canonical(actual))
        for name in ('source_id', 'model_id'):
            object.__setattr__(self, name, actual[name])
        object.__setattr__(self, 'package_directory', Path(actual['package_directory']))
        object.__setattr__(self, 'packages', MappingProxyType({k: Path(v) for k, v in actual['packages'].items()}))
        object.__setattr__(self, 'module_files', MappingProxyType({k: Path(v['path']) for k, v in actual['modules'].items()}))
        object.__setattr__(self, 'module_sha256', MappingProxyType({k: v['sha256'] for k, v in actual['modules'].items()}))
        object.__setattr__(self, 'asset_files', MappingProxyType({v['path']: v['sha256'] for v in actual['assets'].values()}))
        for name, relative in _DESCRIPTION.items():
            object.__setattr__(self, name, Path(actual['assets']['erc_description/' + relative]['path']))
        object.__setattr__(self, '_sealed', True)
        self.verify_import_origins()
        return self

    def __setattr__(self, name, value):
        raise AttributeError('installed geometry identity is immutable')

    def __delattr__(self, name):
        raise AttributeError('installed geometry identity is immutable')

    def to_descriptor(self):
        return json.loads(self._descriptor)

    def verify_import_origins(self):
        """Fail if any loaded package module came from a different install."""
        found = set()
        descriptor = self.to_descriptor()
        for name, module in tuple(sys.modules.items()):
            if name != PACKAGE and not name.startswith(PACKAGE + '.'):
                continue
            if module is None:
                raise ValueError('incomplete package import: ' + name)
            expected = self.module_files.get(name)
            filename = getattr(module, '__file__', None)
            origin = getattr(getattr(module, '__spec__', None), 'origin', None)
            if expected is None or filename is None or origin is None:
                raise ValueError('unmapped installed import: ' + name)
            if Path(filename).resolve(strict=True) != expected or Path(origin).resolve(strict=True) != expected:
                raise ValueError('mixed installed geometry import origin: ' + name)
            if hasattr(module, '__path__'):
                relative = descriptor['modules'][name]['relative']
                expected_dir = (self.package_directory / relative).parent.resolve(strict=True)
                if tuple(Path(p).resolve(strict=True) for p in module.__path__) != (expected_dir,):
                    raise ValueError('mixed installed package search path: ' + name)
            found.add(name)
        if PACKAGE not in found or PACKAGE + '.installed_geometry_identity' not in found:
            raise ValueError('identity must be imported from the admitted package')

    def verify(self):
        expected = self.to_descriptor()
        if _capture(self.package_directory, dict(self.packages)) != expected:
            raise ValueError('installed geometry changed since capture')
        self.verify_import_origins()

    def verify_scene_models(self, bin_scene, table_scene):
        """Original bin/table model gates; geometric scene validation is separate."""
        if not isinstance(bin_scene, dict) or _file(self.bin_mesh)['sha256'] != bin_scene.get('mesh_sha256'):
            raise RuntimeError('placement_scene_bin_model_changed')
        if (not isinstance(table_scene, dict) or table_scene.get('mesh_sha256') != _TABLE_MESH
                or _file(self.table_mesh)['sha256'] != _TABLE_MESH
                or _file(self.table_sdf)['sha256'] not in _TABLE_SDFS):
            raise RuntimeError('placement_scene_table_model_changed')
