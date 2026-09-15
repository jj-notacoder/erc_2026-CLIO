import math
import unittest
from unittest.mock import patch
import numpy as np
from erc_phase1_solution import bin_scene as m

def fixture(yaw=.35,noise=0.,include=('floor','left','right','rear'),width=.29):
    x=np.linspace(-.135,.135,24);y=np.linspace(-.085,.095,18);z=np.linspace(-.24,.23,30)
    pts={
      'floor':np.asarray([[a,-.095,b] for a in x for b in z]),
      'left':np.asarray([[-width/2,a,b] for a in y for b in z]),
      'right':np.asarray([[width/2,a,b] for a in y for b in z]),
      'rear':np.asarray([[a,b,.245] for a in x for b in y])}
    local=np.concatenate([pts[k] for k in include])
    rng=np.random.default_rng(33);local+=rng.uniform(-noise,noise,local.shape)
    long=np.array([math.cos(yaw),math.sin(yaw),0.])
    up=np.array([0.,0.,1.]);side=np.cross(up,long)
    r=np.column_stack((side,up,long));origin=np.array([1.3,-.2,.81])
    return local@r.T+origin,origin,r,origin-.7*long+.6*up

class BinSceneTests(unittest.TestCase):
    def test_recovers_translated_rotated_inner_planes_without_center_bounds(self):
        p,o,r,c=fixture(noise=.0002)
        out=m.fit_bin_points(p,c)
        self.assertTrue(out['valid'],out)
        np.testing.assert_allclose(out['origin'],o,atol=.0003)
        np.testing.assert_allclose(out['rotation'],r,atol=.001)
        np.testing.assert_allclose(out['floor_center'],o+r@[0,-.095,0],atol=.0003)
        self.assertFalse(out['quality']['partial_cloud_bounds_used_for_center'])

    def test_missing_any_required_visible_feature_rejects(self):
        for missing in ('floor','left','right','rear'):
            with self.subTest(missing=missing):
                p,_,_,c=fixture(include=tuple(k for k in ('floor','left','right','rear') if k!=missing))
                self.assertFalse(m.fit_bin_points(p,c)['valid'])

    def test_outer_sides_cannot_be_relabeled_inner_sides(self):
        p,_,_,c=fixture(width=.31)
        self.assertFalse(m.fit_bin_points(p,c)['valid'])

    def test_rear_seen_from_outside_cannot_be_inner_wall(self):
        p,o,r,_=fixture()
        self.assertFalse(m.fit_bin_points(p,o+.7*r[:,2]+.6*r[:,1])['valid'])

    def test_partial_cloud_translation_keeps_visible_planes_as_constraints(self):
        # Clip observed floor at a non-central position; inferred center must
        # remain set by known wall coordinates rather than floor-cloud bounds.
        p,o,r,c=fixture()
        local=(p-o)@r
        p=p[(local[:,1]>-.09)|(local[:,2]>.02)]
        out=m.fit_bin_points(p,c)
        self.assertTrue(out['valid'],out)
        np.testing.assert_allclose(out['origin'],o,atol=1e-6)

    def test_nonfinite_or_insufficient_points_reject(self):
        for p in (np.zeros((20,3)),np.full((700,3),np.nan)):
            self.assertFalse(m.fit_bin_points(p,[0,0,1])['valid'])

    def test_large_red_object_merger_rejects_finite_surface_consistency(self):
        p,o,r,c=fixture()
        rogue=np.random.default_rng(8).uniform([-.1,.02,-.1],[.1,.09,.1],(1000,3))@r.T+o
        self.assertFalse(m.fit_bin_points(np.vstack((p,rogue)),c)['valid'])

    def test_table_only_checks_vertical_consistency_never_horizontal_center(self):
        p,o,r,c=fixture()
        floor=o+r@[0,-.095,0]
        table={'valid':True,'frame':'base_footprint','model':'erc_table_two_edge_rgbd_v1',
               'origin':[7.,-8.,floor[2]-.01],'rotation':np.eye(3).tolist()}
        a=m.fit_bin_points(p,c,table_scene=table)
        self.assertTrue(a['valid'],a)
        np.testing.assert_allclose(a['origin'],o,atol=1e-6)
        table['origin'][2]+=.08
        self.assertFalse(m.fit_bin_points(p,c,table_scene=table)['valid'])

    def test_optional_table_requires_matching_frame_model_and_rigid_up(self):
        p,o,r,c=fixture();floor=o+r@[0,-.095,0]
        base={'valid':True,'frame':'base_footprint','model':'erc_table_two_edge_rgbd_v1',
              'origin':[0.,0.,floor[2]-.01],'rotation':np.eye(3).tolist()}
        for change in ({'frame':'odom'},{'model':'unknown'},{'valid':1},
                       {'rotation':np.diag([1,1,-1]).tolist()}):
            with self.subTest(change=change):
                self.assertFalse(m.fit_bin_points(p,c,table_scene={**base,**change})['valid'])

    def test_registration_margin_adds_translation_and_rotation_extent(self):
        p,_,_,c=fixture();a=m.fit_bin_points(p,c)
        self.assertTrue(a['valid'])
        self.assertEqual(a['modeled_margin_m'],.005)
        self.assertGreater(a['registration_margin_m'],a['translation_error_m'])
        self.assertGreaterEqual(a['translation_error_m'],.005)
        self.assertEqual(a['rotation_error_rad'],.01)
        self.assertTrue(a['quality']['uncertainty_is_engineering_assumption'])

    def test_two_distinct_accepted_registrations_reject_ambiguity(self):
        p,_,_,c=fixture();groups=m._planes(p)
        extra=[]
        for g in groups:
            extra.append(dict(g,center=g['center']+[.1,0,0],points=g['points']+[.1,0,0]))
        # Isolate uniqueness policy: geometry consistency is separately tested.
        with patch.object(m,'_planes',return_value=groups+extra),patch.object(m,'_surface_residual',side_effect=lambda x:np.zeros(len(x))):
            self.assertEqual(m.fit_bin_points(p,c)['reason'],'ambiguous_cad_registration')

    def test_rgbd_entry_rejects_invalid_frame_or_clipped_box(self):
        image=np.zeros((30,40,3),np.uint8);depth=np.ones((30,40));k=np.array([[20,0,20],[0,20,15],[0,0,1.]])
        self.assertFalse(m.fit_bin_scene(image,depth,k,np.eye(4),{'bbox':[0,2,20,20]})['valid'])
        tf=np.eye(4);tf[0,0]=2
        self.assertFalse(m.fit_bin_scene(image,depth,k,tf,{'bbox':[4,4,20,20]})['valid'])

if __name__=='__main__':unittest.main(verbosity=2)
