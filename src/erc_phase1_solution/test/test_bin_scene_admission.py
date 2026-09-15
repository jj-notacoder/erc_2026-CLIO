"""Pure scene/epoch protocol tests; no ROS, model load, or transport mocks."""
import copy
import math
import unittest

import numpy as np

from erc_phase1_solution.bin_scene_admission import (
    BIN_MODEL, BIN_MESH_SHA256, CAD_RADIUS, OUTER_BOUNDS, CAVITY_BOUNDS,
    validate_bin_scene, match_registered_scenes,
)
from erc_phase1_solution.placement_scene_context import matching_table_scene


def fixture():
    yaw=.37
    rotation=np.array([[-math.sin(yaw),0.,math.cos(yaw)],
                       [math.cos(yaw),0.,math.sin(yaw)],[0.,1.,0.]])
    origin=np.array([.982,.044,.845])
    residual=.001
    scene=dict(valid=True,model=BIN_MODEL,frame='base_footprint',mesh_sha256=BIN_MESH_SHA256,
        origin=origin.tolist(),rotation=rotation.tolist(),
        floor_center=(origin+rotation@np.array([0.,-.095,0.])).tolist(),
        floor_point_model=[0.,-.095,0.],outer_bounds=[list(v) for v in OUTER_BOUNDS],
        cavity_bounds=[list(v) for v in CAVITY_BOUNDS],modeled_margin_m=.005,
        translation_error_m=.005+residual,rotation_error_rad=.01,
        registration_margin_m=.005+residual+2*CAD_RADIUS*math.sin(.01/2),
        quality=dict(floor_rear_registration_observed=True,accepted_candidates=2,
            finite_cad_coverage=.98,finite_surface_residual_p95_m=residual,
            feature_indices=[0,1,2,3]))
    stamp=1_000_000_000
    entry=dict(observation_stamp_ns=stamp,bin_floor_point_base=[.89,.007,.751],
        table_scene=dict(valid=True,frame='base_footprint',origin=[.98,.04,.74],
            rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),bin_scene=scene,
        scene_base_reference=dict(frame_id='odom',child_frame_id='base_footprint',
            requested_stamp_ns=stamp,producer_stamp_ns=stamp,pose=[.1,-.2,.3]))
    context=dict(stamp_ns=1_200_000_000,base_pose=[.1,-.2,.3],parked_joints={'head_1_joint':0.},
        odom_frame_id='odom',odom_child_frame_id='base_footprint',odom_producer_stamp_ns=1_190_000_000)
    return scene,entry,context


class BinSceneAdmissionTests(unittest.TestCase):
    def setUp(self):self.scene,self.entry,self.context=fixture()
    def match(self):return match_registered_scenes(self.entry,1_000_000_000,[.89,.007,.751],self.context)
    def test_nonidentity_scene_and_distinct_legacy_point_pass(self):
        table,scene=self.match()
        self.assertGreater(np.linalg.norm(np.array(scene['floor_center'])-self.entry['bin_floor_point_base']),.09)
        np.testing.assert_array_equal(scene['rotation'],self.scene['rotation'])
        self.assertEqual(table,self.entry['table_scene'])
    def test_complete_nested_copy_is_independent_both_ways(self):
        table,scene=self.match()
        self.entry['table_scene']['origin'][0]=5
        self.scene['rotation'][0][0]=5
        self.scene['quality']['feature_indices'][0]=9
        self.assertNotEqual(table['origin'][0],5)
        self.assertNotEqual(scene['rotation'][0][0],5)
        self.assertEqual(scene['quality']['feature_indices'][0],0)
        scene['floor_center'][0]=8
        self.assertNotEqual(self.scene['floor_center'][0],8)
    def test_default_matcher_needs_no_new_metadata(self):
        self.entry.pop('bin_scene');self.entry.pop('scene_base_reference')
        self.assertIs(matching_table_scene(self.entry,1_000_000_000,[.89,.007,.751]),self.entry['table_scene'])
    def test_invalid_and_unrecognized_model_or_frame_rejected(self):
        for field,value in [('valid',False),('valid',1),('model','other'),('mesh_sha256','bad'),('frame','odom')]:
            with self.subTest(field=field,value=value):
                s=copy.deepcopy(self.scene);s[field]=value
                with self.assertRaises(RuntimeError):validate_bin_scene(s)
    def test_all_geometry_arrays_require_finite_exact_shape(self):
        for field in ['origin','rotation','floor_center','floor_point_model','outer_bounds','cavity_bounds']:
            for value in [None,[],[math.nan],[[math.inf]]]:
                with self.subTest(field=field,value=value):
                    s=copy.deepcopy(self.scene);s[field]=value
                    with self.assertRaises(RuntimeError):validate_bin_scene(s)
    def test_reflection_scale_and_downward_up_rejected(self):
        for transform in [np.diag([-1.,1.,1.]),np.diag([1.001,1.,1.]),np.diag([1.,-1.,-1.])]:
            with self.subTest(transform=transform.tolist()):
                s=copy.deepcopy(self.scene);s['rotation']=(np.array(s['rotation'])@transform).tolist()
                with self.assertRaisesRegex(RuntimeError,'rotation'):validate_bin_scene(s)
    def test_tilt_beyond_shared_five_degree_target_limit_rejected(self):
        a=math.radians(5.01);tilt=np.array([[1.,0.,0.],[0.,math.cos(a),-math.sin(a)],[0.,math.sin(a),math.cos(a)]])
        self.scene['rotation']=(tilt@np.array(self.scene['rotation'])).tolist()
        with self.assertRaisesRegex(RuntimeError,'rotation'):validate_bin_scene(self.scene)
    def test_derived_floor_mismatch_rejected(self):
        self.scene['floor_center'][0]+=.0001
        with self.assertRaisesRegex(RuntimeError,'floor_origin'):validate_bin_scene(self.scene)
    def test_cad_space_cannot_be_enlarged_or_shifted(self):
        for field,index in [('outer_bounds',(1,0)),('cavity_bounds',(1,2)),('floor_point_model',(1,))]:
            with self.subTest(field=field):
                s=copy.deepcopy(self.scene)
                if len(index)==2:s[field][index[0]][index[1]]+=1e-8
                else:s[field][index[0]]+=1e-8
                with self.assertRaisesRegex(RuntimeError,'cad_bounds'):validate_bin_scene(s)
    def test_sign_ambiguity_or_missing_rear_evidence_rejected(self):
        for field,value in [('floor_rear_registration_observed',False),('accepted_candidates',0),
                            ('accepted_candidates',True),('ambiguous_cad_registration',True)]:
            with self.subTest(field=field):
                s=copy.deepcopy(self.scene);s['quality'][field]=value
                with self.assertRaises(RuntimeError):validate_bin_scene(s)
    def test_quality_and_margin_version_contract(self):
        changes=[('finite_cad_coverage',.9499),('finite_cad_coverage',1.001),
                 ('finite_surface_residual_p95_m',-.001),('finite_surface_residual_p95_m',.00301)]
        for field,value in changes:
            with self.subTest(field=field,value=value):
                s=copy.deepcopy(self.scene);s['quality'][field]=value
                with self.assertRaises(RuntimeError):validate_bin_scene(s)
        for field,value in [('modeled_margin_m',.004999),('translation_error_m',.0059),
            ('translation_error_m',.008001),('rotation_error_rad',.00999),('rotation_error_rad',.02),
            ('registration_margin_m',.009),('registration_margin_m',.12)]:
            with self.subTest(field=field,value=value):
                s=copy.deepcopy(self.scene);s[field]=value
                with self.assertRaises(RuntimeError):validate_bin_scene(s)
    def test_nonfinite_or_boolean_margin_never_admitted(self):
        for field in ['modeled_margin_m','registration_margin_m','translation_error_m','rotation_error_rad']:
            for value in [math.inf,math.nan,-math.inf,True,'0.01']:
                with self.subTest(field=field,value=value):
                    s=copy.deepcopy(self.scene);s[field]=value
                    with self.assertRaises(RuntimeError):validate_bin_scene(s)
    def test_translation_covers_fit_residual_and_registration_covers_rotation(self):
        self.scene['quality']['finite_surface_residual_p95_m']=.003
        self.scene['translation_error_m']=.008
        self.scene['registration_margin_m']=.008+2*CAD_RADIUS*math.sin(.005)
        self.assertEqual(validate_bin_scene(self.scene),self.scene)
        self.scene['registration_margin_m']-=1e-9
        with self.assertRaisesRegex(RuntimeError,'margin'):validate_bin_scene(self.scene)
    def test_same_entry_table_and_raw_point_must_match(self):
        for action in [lambda e:e.update(observation_stamp_ns=999_000_000),
                       lambda e:e['table_scene'].update(valid=False),
                       lambda e:e['table_scene'].update(frame='odom'),
                       lambda e:e.update(bin_floor_point_base=[.982,.044,.75])]:
            with self.subTest(action=action):
                _,self.entry,self.context=fixture();action(self.entry)
                with self.assertRaises(RuntimeError):self.match()
    def test_missing_selected_bin_or_reference_cannot_reuse_previous_scene(self):
        for key in ['bin_scene','scene_base_reference']:
            with self.subTest(key=key):
                _,self.entry,self.context=fixture();self.entry.pop(key)
                with self.assertRaises(RuntimeError):self.match()
    def test_exact_requested_and_returned_transform_epochs(self):
        for key in ['requested_stamp_ns','producer_stamp_ns']:
            for value in [0,True,1_000_000_001,1_000_000_000.0]:
                with self.subTest(key=key,value=value):
                    _,self.entry,self.context=fixture();self.entry['scene_base_reference'][key]=value
                    with self.assertRaises(RuntimeError):self.match()
    def test_source_and_current_odometry_frames_are_explicit(self):
        for location,key in [('scene_base_reference','frame_id'),('scene_base_reference','child_frame_id'),
                             ('context','odom_frame_id'),('context','odom_child_frame_id')]:
            for value in [None,'map','base_link']:
                with self.subTest(location=location,key=key,value=value):
                    _,self.entry,self.context=fixture()
                    target=self.context if location=='context' else self.entry[location];target[key]=value
                    with self.assertRaisesRegex(RuntimeError,'frame'):self.match()
    def test_raw_odometry_requires_positive_nonfuture_350ms_producer(self):
        for value in [0,True,1_200_000_001,849_999_999]:
            with self.subTest(value=value):
                _,self.entry,self.context=fixture();self.context['odom_producer_stamp_ns']=value
                with self.assertRaises(RuntimeError):self.match()
        self.context['odom_producer_stamp_ns']=850_000_000
        self.match()
    def test_depth_requires_nonfuture_500ms_epoch(self):
        for now in [999_999_999,1_500_000_001]:
            with self.subTest(now=now):
                self.context['stamp_ns']=now;self.context['odom_producer_stamp_ns']=now
                with self.assertRaisesRegex(RuntimeError,'observation_not_fresh'):self.match()
        self.context['stamp_ns']=1_500_000_000;self.context['odom_producer_stamp_ns']=1_500_000_000
        self.match()
    def test_translation_is_euclidean_not_independent_axis_allowance(self):
        self.context['base_pose']=[.1015,-.1985,.3]
        with self.assertRaisesRegex(RuntimeError,'base_moved'):self.match()
        self.context['base_pose']=[.1019,-.2,.3];self.match()
    def test_yaw_wrap_and_original_limit(self):
        self.entry['scene_base_reference']['pose'][2]=math.pi-.001
        self.context['base_pose'][2]=-math.pi+.001
        self.match()
        self.context['base_pose'][2]=-math.pi+.005
        with self.assertRaisesRegex(RuntimeError,'base_moved'):self.match()
    def test_invalid_pose_cannot_bypass_norm_checks(self):
        for value in [[math.nan,0.,0.],[0.,0.],[0.,0.,math.inf]]:
            with self.subTest(value=value):
                _,self.entry,self.context=fixture();self.context['base_pose']=value
                with self.assertRaises(RuntimeError):self.match()
    def test_snapshot_replacement_does_not_change_returned_bundle(self):
        table,scene=self.match()
        _,new_entry,_=fixture();new_entry['bin_scene']['origin'][0]=42
        self.entry.clear();self.entry.update(new_entry)
        self.assertNotEqual(scene['origin'][0],42)
        self.assertEqual(table['origin'],[.98,.04,.74])


if __name__ == '__main__':unittest.main()
