"""Small standard-library tests; synthetic evidence, no ROS/robot execution."""
import unittest
from erc_phase1_solution.release_evidence import AttemptIdentity,ReleaseEvidence,measured_pose_is_open


class ReleaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.identity=AttemptIdentity('trial1','attempt1','book_col_3_row_2_red')
        self.e=ReleaseEvidence(self.identity)
        self.fields=vars(self.identity)
        self.start=1_000_000_000

    def opened(self):
        self.assertTrue(self.e.opening(dict(self.fields,event='placement_open_measured',
            verified=True,producer_stamp_ns=self.start),self.start))

    def returned(self,now):
        self.assertTrue(self.e.hand_return(dict(self.fields,event='placement_hand_return_measured',
            verified=True,producer_stamp_ns=now),now))
        self.e.motion_terminal(dict(self.fields,event='succeeded',command='place'),now,400.)

    def test_pre_release_and_late_old_contacts_cannot_qualify(self):
        for offset in range(0,600_000_001,2_000_000):
            self.e.contact(self.start-600_000_000+offset,self.start,exact_target_bin_pair=True)
        self.opened()
        self.e.contact(self.start,self.start+20_000_000,exact_target_bin_pair=True)
        self.assertFalse(self.e.evaluate(self.start+20_000_000,1.)['post_release_contact_verified'])

    def test_10hz_and500hz_contact_intervals_work_without_containment_claim(self):
        for period in (100_000_000,2_000_000):
            with self.subTest(period=period):
                self.e=ReleaseEvidence(self.identity);self.opened()
                for delta in range(period,700_000_001,period):
                    now=self.start+delta
                    self.e.contact(now,now,exact_target_bin_pair=True)
                self.returned(now)
                result=self.e.evaluate(now,400.)
                self.assertEqual(result['status'],'operation_completed')
                self.assertTrue(result['operation_completed'])
                self.assertEqual(result['success_scope'],'placement_operation_completed')
                self.assertEqual(result['delivery_outcome'],'released_with_bin_contact')
                self.assertIsNone(result['physical_inside_verified'])
                self.assertTrue(result['post_release_contact_verified'])
                self.assertFalse(result['containment_verified'])
                self.assertFalse(result['physical_delivery_verified'])
                self.assertFalse(result['hand_clear_verified'])
                self.assertLessEqual(len(self.e.contact_stamps),351)

    def test_duplicates_wrong_pairs_and_future_frames_never_create_span(self):
        self.opened()
        for _ in range(500):
            self.e.contact(self.start+1,self.start+1,exact_target_bin_pair=True)
            self.e.contact(self.start+500_000_000,self.start+1,exact_target_bin_pair=True)
            self.e.contact(self.start+2,self.start+1,exact_target_bin_pair=False)
        result=self.e.evaluate(self.start+1,1.)
        self.assertFalse(result['post_release_contact_verified'])
        self.assertEqual(result['contact_samples'],1)

    def test_contact_before_open_status_delivery_is_filtered_by_producer_epoch(self):
        self.e.contact(self.start+20_000_000,self.start+20_000_000,exact_target_bin_pair=True)
        self.assertTrue(self.e.opening(dict(self.fields,event='placement_open_measured',verified=True,
            producer_stamp_ns=self.start),self.start+20_000_000))
        self.assertEqual(self.e.contact_stamps,{self.start+20_000_000})

    def test_future_open_epoch_waits_for_clock_without_becoming_release_proof(self):
        payload=dict(self.fields,event='placement_open_measured',verified=True,
                     producer_stamp_ns=self.start+2_000_000)
        self.assertFalse(self.e.opening(payload,self.start))
        self.assertFalse(self.e.evaluate(self.start,1.)['measured_open'])
        self.assertTrue(self.e.evaluate(self.start+2_000_000,1.01)['measured_open'])

    def test_old_attempt_cannot_set_open_return_or_terminal(self):
        payload=dict(self.fields,placement_attempt_id='old',event='placement_open_measured',
                     verified=True,producer_stamp_ns=self.start)
        self.assertFalse(self.e.opening(payload,self.start))
        self.assertFalse(self.e.hand_return(payload,self.start))
        self.assertFalse(self.e.motion_terminal(dict(payload,event='succeeded',command='place'),self.start,5.))
        self.assertIsNone(self.e.verification_start)

    def test_verification_deadline_starts_after_motion_not_at_dispatch_and_never_renews(self):
        # Minutes of planning are not a verification period.
        self.assertEqual(self.e.evaluate(self.start,399.)['status'],'pending')
        self.opened();self.returned(self.start+100_000_000)
        initial=self.e.verification_start
        self.e.motion_terminal(dict(self.fields,event='succeeded',command='place'),self.start+200_000_000,405.)
        self.assertEqual(self.e.verification_start,initial)
        self.assertEqual(self.e.evaluate(self.start+500_000_000,431.)['status'],'expired')

    def test_gap_or_stale_final_contact_removes_previous_qualification(self):
        self.opened()
        for delta in range(2_000_000,602_000_000,2_000_000):
            self.e.contact(self.start+delta,self.start+delta,exact_target_bin_pair=True)
        self.assertTrue(self.e.evaluate(self.start+600_000_000,1.)['post_release_contact_verified'])
        self.assertFalse(self.e.evaluate(self.start+800_000_000,1.2)['post_release_contact_verified'])
        self.e.contact(self.start+800_000_000,self.start+800_000_000,exact_target_bin_pair=True)
        self.assertFalse(self.e.evaluate(self.start+800_000_000,1.2)['post_release_contact_verified'])

    def test_terminal_without_measured_open_still_expires_on_both_deadlines(self):
        for ros_delta,wall_delta in ((10_000_000_001,1.),(1,30.000001)):
            with self.subTest(ros_delta=ros_delta):
                self.e=ReleaseEvidence(self.identity)
                self.assertTrue(self.e.motion_terminal(dict(self.fields,event='succeeded',command='place'),
                    self.start,400.))
                result=self.e.evaluate(self.start+ros_delta,400.+wall_delta)
                self.assertEqual(result['status'],'expired')
                self.assertEqual(result['reason'],'verification_deadline_expired')
                self.assertFalse(result['measured_open'])
                self.assertFalse(result['operation_completed'])

    def test_terminal_without_measured_return_still_expires_and_never_renews(self):
        self.opened()
        payload=dict(self.fields,event='succeeded',command='place')
        self.assertTrue(self.e.motion_terminal(payload,self.start,400.))
        self.assertTrue(self.e.motion_terminal(payload,self.start+1,420.))
        self.assertEqual(self.e.verification_start,(self.start,400.))
        result=self.e.evaluate(self.start+2,430.000001)
        self.assertEqual(result['status'],'expired')
        self.assertFalse(result['hand_return_measured'])
        self.assertFalse(result['operation_completed'])

    def test_missing_open_does_not_mask_invalid_wall_clock(self):
        self.assertTrue(self.e.motion_terminal(dict(self.fields,event='succeeded',command='place'),
            self.start,400.))
        result=self.e.evaluate(self.start+1,float('nan'))
        self.assertEqual(result['status'],'invalid')
        self.assertEqual(result['reason'],'invalid_wall_clock')

    def test_overflow_fails_closed(self):
        self.opened()
        for index in range(1025):
            stamp=self.start+index+1
            self.e.contact(stamp,stamp,exact_target_bin_pair=True)
        self.assertEqual(self.e.fault,'contact_history_rate_overflow')

    def test_raw_open_feedback_not_command_success(self):
        names=('torso_lift_joint','arm_left_1_joint')
        raw=dict(producer_stamp_ns=self.start,
            positions=dict(torso_lift_joint=.1,arm_left_1_joint=.2,gripper_left_finger_joint=.069),
            velocities=dict(torso_lift_joint=0.,arm_left_1_joint=0.,gripper_left_finger_joint=0.))
        odom=dict(stamp_ns=self.start,linear_speed=0.,angular_speed=0.)
        self.assertTrue(measured_pose_is_open(raw,odom,(.1,.2),self.start,names)[0])
        for fault in ('closed','moving','future','missing_velocity'):
            with self.subTest(fault=fault):
                candidate=dict(raw,positions=dict(raw['positions']),velocities=dict(raw['velocities']))
                if fault=='closed':candidate['positions']['gripper_left_finger_joint']=.018
                if fault=='moving':candidate['velocities']['arm_left_1_joint']=.02
                if fault=='future':candidate['producer_stamp_ns']+=2_000_000
                if fault=='missing_velocity':del candidate['velocities']['arm_left_1_joint']
                self.assertFalse(measured_pose_is_open(candidate,odom,(.1,.2),self.start,names)[0])


if __name__=='__main__':unittest.main()
