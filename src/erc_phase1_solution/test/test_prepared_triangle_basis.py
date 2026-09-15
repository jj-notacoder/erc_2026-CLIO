"""Exact immutable facet-basis reuse against the frozen mesh predicates."""
import unittest
from types import SimpleNamespace
import numpy as np
from erc_phase1_solution import kinematics as candidate
from data.reference_mesh_predicates_run17 import reference_triangle_meshes_intersect

baseline=SimpleNamespace(triangle_meshes_intersect=reference_triangle_meshes_intersect)

class TestBasis(unittest.TestCase):
    def test_exact_basis_bytes_and_immutability(self):
        rng=np.random.default_rng(230911)
        mesh=rng.normal(size=(25,3,3))
        mesh[0]=0.; mesh[1,0,0]=-0.; mesh[2,1]=mesh[2,0]
        prepared=candidate.PreparedTriangleMesh(mesh)
        self.assertIsNone(prepared._facet_basis)
        edges,normals=prepared.facet_basis
        for i,triangle in enumerate(mesh):
            expected=np.roll(triangle,-1,axis=0)-triangle
            self.assertEqual(edges[i].tobytes(),expected.tobytes())
            self.assertEqual(normals[i].tobytes(),np.cross(expected[0],expected[1]).tobytes())
        mesh[:]=100
        self.assertIs(prepared.facet_basis,prepared.facet_basis)
        for array in (edges,normals):
            with self.assertRaises(ValueError):array.setflags(write=True)

    def test_each_world_snapshot_has_its_own_basis(self):
        a=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        b=a.copy();b[0,2,2]=np.nextafter(0.,1.)
        p,q=candidate.PreparedTriangleMesh(a),candidate.PreparedTriangleMesh(b)
        self.assertIsNot(p.facet_basis,q.facet_basis)
        self.assertNotEqual(p.facet_basis[0].tobytes(),q.facet_basis[0].tobytes())

    def test_whole_mesh_separation_leaves_basis_lazy(self):
        a=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        p,q=candidate.PreparedTriangleMesh(a),candidate.PreparedTriangleMesh(a+100.)
        self.assertFalse(candidate.triangle_meshes_intersect(p,q))
        self.assertIsNone(p._facet_basis);self.assertIsNone(q._facet_basis)

    def equivalent(self,a,b,tolerance=1e-9,closed=(False,False)):
        for left,right,flags in ((a,b,closed),(b,a,closed[::-1])):
            kwargs=dict(first_watertight=flags[0],second_watertight=flags[1])
            expected=baseline.triangle_meshes_intersect(left,right,tolerance,**kwargs)
            p,q=candidate.PreparedTriangleMesh(left),candidate.PreparedTriangleMesh(right)
            for x,y in ((left,right),(p,right),(left,q),(p,q)):
                self.assertEqual(candidate.triangle_meshes_intersect(x,y,tolerance,**kwargs),expected)

    def test_filtered_indices_swapped_order_and_degeneracy(self):
        rng=np.random.default_rng(231)
        for index in range(15):
            a=rng.normal(size=(1+index%4,3,3)); b=rng.normal(size=(1+index%7,3,3))
            if index%3==0:a[0,1:]=a[0,0]
            if index%4==0:b[:,:,2]=a[0,0,2]
            # Prepend rejected facets to exercise original source indexing.
            a=np.concatenate((a+20.,a))
            self.equivalent(a,b,(0.,1e-9,1e-6)[index%3])

    def test_touch_and_adjacent_float_tolerance(self):
        a=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        tolerance=1e-9
        for d in (0.,np.nextafter(tolerance,0.),tolerance,np.nextafter(tolerance,np.inf)):
            self.equivalent(a,a+[0.,0.,d],tolerance)

    def test_kernel_receives_exact_bases_after_filter_and_size_swap(self):
        triangle=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        a=np.stack((triangle*3.+30.,triangle*4.+40.,triangle*5.+50.,triangle*6.+60.,triangle*2.))
        b=np.stack((triangle*7.+20.,triangle*.4+[.1,.1,0.]))
        original=candidate._triangles_intersect_with_basis
        calls=[]
        def spy(left,right,edges_left,edges_right,normal_left,normal_right,tolerance):
            for actual,edges,normal in ((left,edges_left,normal_left),(right,edges_right,normal_right)):
                expected=np.roll(actual,-1,axis=0)-actual
                self.assertEqual(edges.tobytes(),expected.tobytes())
                self.assertEqual(normal.tobytes(),np.cross(expected[0],expected[1]).tobytes())
            calls.append((left.copy(),right.copy()))
            return original(left,right,edges_left,edges_right,normal_left,normal_right,tolerance)
        candidate._triangles_intersect_with_basis=spy
        try:
            self.assertTrue(candidate.triangle_meshes_intersect(candidate.PreparedTriangleMesh(a),candidate.PreparedTriangleMesh(b)))
        finally:
            candidate._triangles_intersect_with_basis=original
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0].tobytes(),b[1].tobytes())
        self.assertEqual(calls[0][1].tobytes(),a[4].tobytes())

    def test_contained_disconnected_closed_mesh(self):
        vertices=np.array([(x,y,z) for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        box=vertices[faces]
        disconnected=np.concatenate((box*.1+[4.,0.,0.],box*.1))
        for flags in ((True,False),(False,True),(True,True)):
            self.equivalent(box,disconnected,closed=flags)

    def test_empty_and_invalid_inputs_retain_validation(self):
        a=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        for empty in ([],np.empty((0,3,3)),np.empty((1,0))):self.equivalent(empty,a)
        for bad in (np.zeros((2,3)),np.full((1,3,3),np.nan)):
            with self.assertRaises(ValueError):candidate.PreparedTriangleMesh(bad)
        with self.assertRaises(ValueError):candidate.triangle_meshes_intersect(candidate.PreparedTriangleMesh(a),candidate.PreparedTriangleMesh(a),-1.)

if __name__=='__main__':unittest.main(verbosity=2)
