"""Portable file/origin tests; synthetic install layouts, no ROS or geometry."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from erc_phase1_solution import installed_geometry_identity as identity


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def module(name, path, package=False):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    obj.__spec__ = importlib.util.spec_from_file_location(name, path)
    if package:
        obj.__path__ = [str(path.parent)]
    return obj


@pytest.fixture
def installed(tmp_path, monkeypatch):
    package = tmp_path / 'install/lib/python/site-packages/erc_phase1_solution'
    put(package / '__init__.py', '')
    put(package / 'installed_geometry_identity.py', Path(identity.__file__).read_bytes())
    put(package / 'kinematics.py', 'VALUE = 1\n')
    # Model shares intentionally contain synthetic files. These tests exercise
    # identity and closure, not official geometry or current model dimensions.
    description, meshes = tmp_path / 'install/share/erc_description', tmp_path / 'install/share/mesh_package'
    for name, root in [('erc_description', description), ('mesh_package', meshes)]:
        put(root / 'package.xml', '<package><name>' + name + '</name></package>')
    put(description / identity._DESCRIPTION['urdf'], '<robot name="test"><link name="base">'
        '<collision><geometry><mesh filename="package://mesh_package/collision.stl"/></geometry></collision>'
        '<visual><geometry><mesh filename="package://mesh_package/visual.stl"/></geometry></visual>'
        '</link></robot>')
    put(meshes / 'collision.stl', b'collision triangles')
    put(meshes / 'visual.stl', b'visual triangles')
    for key in ('bin_mesh', 'table_mesh'):
        put(description / identity._DESCRIPTION[key], key)
    for key, mesh in [('bin_sdf','bin_mesh'), ('table_sdf','table_mesh')]:
        put(description / identity._DESCRIPTION[key], '<sdf><model><link><collision><geometry><mesh><uri>'
            'model:///erc_description/' + identity._DESCRIPTION[mesh] + '</uri></mesh></geometry>'
            '</collision></link></model></sdf>')
    # Simulate only import metadata at the fresh-child boundary. No synthetic
    # file is executed. The real module's functions remain the code under test.
    for name in tuple(sys.modules):
        if name == identity.PACKAGE or name.startswith(identity.PACKAGE + '.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, identity.PACKAGE, module(identity.PACKAGE, package / '__init__.py', True))
    name = identity.PACKAGE + '.installed_geometry_identity'
    monkeypatch.setitem(sys.modules, name, module(name, package / 'installed_geometry_identity.py'))
    shares = {'erc_description': description, 'mesh_package': meshes}
    return package, shares


def capture(installed):
    package, shares = installed
    return identity.InstalledGeometryIdentity.capture(package_directory=package, package_shares=shares)


def test_normal_install_roundtrip_and_complete_resource_closure(installed):
    current = capture(installed)
    restored = identity.InstalledGeometryIdentity.from_descriptor(json.loads(json.dumps(current.to_descriptor())))
    assert restored.to_descriptor() == current.to_descriptor()
    assert set(current.module_files) == {identity.PACKAGE, identity.PACKAGE + '.installed_geometry_identity', identity.PACKAGE + '.kinematics'}
    assert len(current.asset_files) == 9  # five description files, two package XMLs, two URDF meshes
    assert current.urdf.is_file() and current.bin_sdf.is_file()
    assert not any('test' in key or 'certificate' in key for key in current.module_files)
    current.verify()


def test_descriptor_mutation_cannot_modify_owner(installed):
    current = capture(installed)
    descriptor = current.to_descriptor()
    descriptor['assets'].clear()
    assert current.to_descriptor()['assets']
    with pytest.raises(AttributeError):
        current.source_id = '0' * 64
    with pytest.raises(TypeError):
        current.asset_files['other'] = '0' * 64


@pytest.mark.parametrize('field', ['source_id', 'model_id', 'schema', 'package'])
def test_descriptor_identity_tampering_rejected(installed, field):
    descriptor = capture(installed).to_descriptor()
    descriptor[field] = 'wrong'
    with pytest.raises(ValueError):
        identity.InstalledGeometryIdentity.from_descriptor(descriptor)


@pytest.mark.parametrize('section', ['modules', 'assets', 'packages'])
def test_descriptor_omission_rejected(installed, section):
    descriptor = capture(installed).to_descriptor()
    del descriptor[section][next(iter(descriptor[section]))]
    with pytest.raises((ValueError, KeyError)):
        identity.InstalledGeometryIdentity.from_descriptor(descriptor)


@pytest.mark.parametrize('kind', ['module', 'asset', 'added_module', 'removed_module'])
def test_changed_file_or_inventory_rejected(installed, kind):
    current = capture(installed)
    package, shares = installed
    if kind == 'module':
        put(package / 'kinematics.py', 'VALUE = 2\n')
    elif kind == 'asset':
        put(shares['mesh_package'] / 'collision.stl', 'changed')
    elif kind == 'added_module':
        put(package / 'later.py', '')
    else:
        (package / 'kinematics.py').unlink()
    with pytest.raises(ValueError):
        current.verify()


def test_ignored_cache_test_and_report_files_not_required(installed):
    current = capture(installed)
    package, _ = installed
    put(package / '__pycache__/kinematics.cpython.pyc', b'cache')
    put(package / 'test/fixture.py', 'never imported')
    put(package / 'test_unrelated.py', 'never imported')
    put(package / 'report.json', '{}')
    current.verify()


@pytest.mark.parametrize('kind', ['wrong_file', 'wrong_spec', 'unmapped', 'missing_spec', 'package_path'])
def test_mixed_import_origin_rejected(installed, monkeypatch, tmp_path, kind):
    current = capture(installed)
    package, _ = installed
    name = identity.PACKAGE + '.kinematics'
    obj = module(name, package / 'kinematics.py')
    other = put(tmp_path / 'other.py', 'VALUE = 1\n')
    if kind == 'wrong_file':
        obj.__file__ = str(other)
    elif kind == 'wrong_spec':
        obj.__spec__ = importlib.util.spec_from_file_location(name, other)
    elif kind == 'unmapped':
        name = identity.PACKAGE + '.unknown'
    elif kind == 'missing_spec':
        obj.__spec__ = None
    else:
        sys.modules[identity.PACKAGE].__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, name, obj)
    with pytest.raises(ValueError):
        current.verify_import_origins()


def test_each_new_import_is_checked(installed, monkeypatch):
    current = capture(installed)
    package, _ = installed
    name = identity.PACKAGE + '.kinematics'
    monkeypatch.setitem(sys.modules, name, module(name, package / 'kinematics.py'))
    current.verify_import_origins()


@pytest.mark.parametrize('uri', ['file:///tmp/x.stl', 'package://mesh_package/../x.stl',
    'package://mesh_package//x.stl', 'package://mesh_package/a\\b.stl'])
def test_unsupported_or_escaping_mesh_uris_rejected(installed, uri):
    package, shares = installed
    put(shares['erc_description'] / identity._DESCRIPTION['urdf'],
        '<robot><link><collision><geometry><mesh filename="' + uri + '"/></geometry></collision></link></robot>')
    with pytest.raises(ValueError):
        capture(installed)


def test_missing_referenced_share_and_wrong_declared_name_rejected(installed):
    package, shares = installed
    with pytest.raises(ValueError):
        identity.InstalledGeometryIdentity.capture(package_directory=package, package_shares={'erc_description': shares['erc_description']})
    put(shares['mesh_package'] / 'package.xml', '<package><name>different</name></package>')
    with pytest.raises(ValueError):
        capture(installed)


def test_symlink_module_and_share_file_roundtrip_and_retarget(installed, tmp_path):
    package, shares = installed
    original = package / 'kinematics.py'
    linked = put(tmp_path / 'source/kinematics.py', original.read_bytes())
    original.unlink()
    original.symlink_to(linked)
    collision = shares['mesh_package'] / 'collision.stl'
    mesh = put(tmp_path / 'source/mesh.stl', collision.read_bytes())
    collision.unlink()
    collision.symlink_to(mesh)
    current = capture(installed)
    current.verify()
    assert current.module_files[identity.PACKAGE + '.kinematics'] == linked
    same_bytes_new_path = put(tmp_path / 'other/mesh.stl', mesh.read_bytes())
    collision.unlink()
    collision.symlink_to(same_bytes_new_path)
    with pytest.raises(ValueError):
        current.verify()


def test_symlink_package_directory_supported(installed, tmp_path, monkeypatch):
    package, shares = installed
    alias = tmp_path / 'colcon_symlink/erc_phase1_solution'
    alias.parent.mkdir()
    alias.symlink_to(package, target_is_directory=True)
    name = identity.PACKAGE
    monkeypatch.setitem(sys.modules, name, module(name, alias / '__init__.py', True))
    current = identity.InstalledGeometryIdentity.capture(package_directory=alias, package_shares=shares)
    current.verify()


def test_symlink_init_file_preserves_logical_package_search_path(installed, tmp_path, monkeypatch):
    package, _ = installed
    init = package / '__init__.py'
    linked = put(tmp_path / 'source/__init__.py', init.read_bytes())
    init.unlink()
    init.symlink_to(linked)
    monkeypatch.setitem(sys.modules, identity.PACKAGE, module(identity.PACKAGE, init, True))
    capture(installed).verify()


@pytest.mark.parametrize('which', ['bin', 'table', 'sdf'])
def test_scene_model_gate_retains_all_hash_checks(installed, monkeypatch, which):
    current = capture(installed)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    # Synthetic fixtures replace constants only for a matched-model positive
    # control; production values remain the original model admissions.
    monkeypatch.setattr(identity, '_TABLE_MESH', digest(current.table_mesh))
    monkeypatch.setattr(identity, '_TABLE_SDFS', frozenset((digest(current.table_sdf),)))
    bin_scene, table_scene = {'mesh_sha256': digest(current.bin_mesh)}, {'mesh_sha256': digest(current.table_mesh)}
    current.verify_scene_models(bin_scene, table_scene)
    if which == 'bin':
        bin_scene['mesh_sha256'] = '0' * 64
    elif which == 'table':
        table_scene['mesh_sha256'] = '0' * 64
    else:
        current.table_sdf.write_text('<sdf/>')
    with pytest.raises(RuntimeError):
        current.verify_scene_models(bin_scene, table_scene)


def test_module_and_resource_limits_fail_closed(installed, monkeypatch):
    monkeypatch.setattr(identity, 'MAX_MODULES', 2)
    with pytest.raises(ValueError):
        capture(installed)
    monkeypatch.setattr(identity, 'MAX_MODULES', 512)
    monkeypatch.setattr(identity, 'MAX_ASSETS', 1)
    with pytest.raises(ValueError):
        capture(installed)
