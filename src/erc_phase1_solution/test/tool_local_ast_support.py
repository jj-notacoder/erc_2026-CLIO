"""Narrow asserted inverses for immutable local tool ownership plumbing.

Only test ASTs are copied/edited. No production module is imported or changed.
"""
import ast
import copy


def restore_tool_constructor_keyword(tree):
    result=copy.deepcopy(tree)
    calls=[n for n in ast.walk(result) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='ShelfCradleGeometry']
    assert len(calls)==1
    expected=ast.parse('ShelfCradleGeometry(urdf, get_package_share_directory, immutable_local=True)',mode='eval').body
    assert ast.dump(calls[0],include_attributes=False)==ast.dump(expected,include_attributes=False)
    calls[0].keywords=[]
    return result


def restore_tool_handle_sample(sample):
    result=copy.deepcopy(sample)
    expected_iter=ast.parse('self.tool.local_surfaces(aperture).items()',mode='eval').body
    loops=[n for n in result.body if isinstance(n,ast.For)
           and ast.dump(n.iter,include_attributes=False)==ast.dump(expected_iter,include_attributes=False)]
    assert len(loops)==1
    loop=loops[0]
    expected=ast.parse("""bin_surface = table_surface = surface
if type(self.tool) is ShelfCradleGeometry:
    getter = getattr(self.tool, 'model_local_surface', None)
    local_mesh = getter(surface) if callable(getter) else None
    if local_mesh is not None:
        from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
        if type(local_mesh) is ModelLocalMesh and local_mesh.matches(surface):
            if type(self.obstacle) is NominalBinObstacle:
                bin_surface = local_mesh
            if type(self.table) is TableSceneObstacle:
                table_surface = local_mesh
""").body
    assert len(loop.body)>=4
    assert [ast.dump(n,include_attributes=False) for n in loop.body[:2]]==[ast.dump(n,include_attributes=False) for n in expected]
    loop.body=loop.body[2:]
    expected_tests=(
        'self.obstacle.intersects(bin_surface, hand, self.tool.watertight[link])',
        'self.table is not None and self.table.intersects(table_surface, hand, self.tool.watertight[link])')
    for item,text in zip(loop.body[:2],expected_tests):
        assert isinstance(item,ast.If)
        expected_test=ast.parse(text,mode='eval').body
        assert ast.dump(item.test,include_attributes=False)==ast.dump(expected_test,include_attributes=False)
        calls=[n for n in ast.walk(item.test) if isinstance(n,ast.Call)]
        assert len(calls)==1
        calls[0].args[0]=ast.Name(id='surface',ctx=ast.Load())
    return result
