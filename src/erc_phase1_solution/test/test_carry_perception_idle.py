"""Actual mission/perception methods with in-memory delivery; no robot motion.

Expensive detectors are spied at their actual call boundary. The ROS messages,
CvBridge conversion, mode resets, timestamp admission and mission branches are
real. This does not test detector accuracy or establish a wall-time saving.
"""
from collections import deque
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock
import struct
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import pytest

from erc_phase1_solution import perception_node as perception_module
from erc_phase1_solution.common import encode_event, decode_event
from erc_phase1_solution.mission_manager import MissionManager
from erc_phase1_solution.perception_node import PerceptionNode


def image(kind, stamp_ns):
    msg=Image()
    msg.header.frame_id='depth_optical_frame'
    msg.header.stamp.sec=stamp_ns//1_000_000_000
    msg.header.stamp.nanosec=stamp_ns%1_000_000_000
    msg.height=msg.width=2
    msg.is_bigendian=0
    msg.encoding='bgr8' if kind=='rgb' else '32FC1'
    msg.step=6 if kind=='rgb' else 8
    msg.data=bytes(12) if kind=='rgb' else struct.pack('<4f',1.,1.,1.,1.)
    return msg


@pytest.fixture
def bus(monkeypatch):
    clock=NS(ns=2_000_000_000)
    p=NS(mode='books',target_column=2,target_colour='red',confirmed_book_row=1,
         bridge=CvBridge(),get_clock=lambda:NS(now=lambda:NS(nanoseconds=clock.ns)),
         camera_info=NS(k=[100.,0.,1.,0.,100.,1.,0.,0.,1.]),
         latest_rgb=None,latest_rgb_message=None,latest_depth=None,latest_depth_message=None,
         maximum_frame_skew=.05,maximum_frame_age=.35,maximum_tracking_frame_skew=.04,
         minimum_colour_area=20.,last_tracking_depth_ns=-1,last_processed_ns=-1,
         ready_sent=True,last_waiting_report_wall=time.monotonic(),
         last_bin_consumed_ns=-1,last_bin_depth_ns=-1,last_bin_verified_ns=-1,last_bin_observed_ns=-1,
         bin_table_height=.76,saved_modes=set(),last_tracking_status='',
         bin_tracker=NS(reset=Mock()),target_tracker=NS(reset=Mock(),last_good=None),
         _publish_tracking_status=Mock(),_tracking_unavailable=Mock(),_publish_status=Mock())
    for name in ('marker_history','book_history','bin_history','bin_rgb_frames','bin_depth_frames'):
        setattr(p,name,deque(maxlen=24))
    for name in ('_on_mode','_on_rgb','_on_depth','_ready','_process','_process_books','_process_bin','_invalidate_bin'):
        setattr(p,name,MethodType(getattr(PerceptionNode,name),p))
    p.tf_buffer=NS(lookup_transform=Mock(return_value=NS(transform=NS(
        translation=NS(x=0.,y=0.,z=0.),rotation=NS(x=0.,y=0.,z=0.,w=1.)))))
    books=Mock(return_value=[])
    bins=Mock(return_value=[])
    monkeypatch.setattr(perception_module,'detect_shelf_books',books)
    monkeypatch.setattr(perception_module,'red_bin_candidates',bins)
    p._on_rgb(image('rgb',1_980_000_000))
    p._on_depth(image('depth',1_980_000_000))

    events=[]
    m=NS(state='PICK',finished=False,wall_started=time.monotonic(),trial_timeout=1000.,
         trial_id='test_current_trial',dry_run=False,delivery_evidence_enabled=False,
         target_physical_column=3,detected_row=1,target_colour='red',target_column=2,
         target_book_model='book_col_3_row_2_red',row_confirmed=True,
         manip_event=None,nav_event=None,pick_attempts=1,max_pick_attempts=2,
         manipulation_timeout=100.,navigation_timeout=100.,perception_timeout=100.,
         empty_head_timing_enabled=False,marker_search_negative_enabled=False,
         head_return_overlap_enabled=False,book_point=object(),bin_point=None,
         bin_candidate=None,bin_invalidated_ns=-1,start_pose=(0.,0.,0.),shelf_normal=(1.,0.),
         get_clock=p.get_clock,_log=Mock(),_elapsed_state=lambda:0.)
    m.mode_pub=NS(topic_name='/erc/perception/mode')
    m.manip_command_pub=NS(topic_name='/erc/manipulation/command')

    def mode_publish(message):
        events.append(('mode',decode_event(message.data)))
        p._on_mode(message)

    def manip_publish(message):
        events.append(('manipulation',decode_event(message.data)))

    m.mode_pub.publish=mode_publish
    m.manip_command_pub.publish=manip_publish
    for name in ('_tick','_command','_perception_mode','_manipulate','_on_manipulation_status',
                 '_expected_target_book_model','_target_identity_confirmed','_manip_succeeded',
                 '_manip_failed','_nav_reached','_nav_failed'):
        setattr(m,name,MethodType(getattr(MissionManager,name),m))

    def state(value,**fields):
        m.state=value
        events.append(('state',value,fields))

    def navigate(*args,**fields):
        events.append(('navigate',args,fields))
        m.nav_event=None

    def abort(reason):
        events.append(('abort',reason))
        m.finished=True

    m._set_state=state;m._navigate=navigate;m._abort=abort
    m._carried_shelf_retreat_goal=lambda:(1.,2.,0.)

    def deliver(event,command='pick',**fields):
        m._on_manipulation_status(String(data=encode_event(event,command=command,**fields)))

    return NS(m=m,p=p,clock=clock,events=events,books=books,bins=bins,deliver=deliver)


def mode_events(bus):
    return [entry[1]['event'] for entry in bus.events if entry[0]=='mode']


def test_successful_pick_idles_after_identity_then_retreats_once(bus):
    b=bus
    b.p._process()
    assert b.books.call_count==1
    b.p.book_history.append('old_book')
    b.p.bin_rgb_frames.append(('old_rgb','old_pixels'))
    b.p.bin_depth_frames.append(('old_depth','old_pixels'))
    b.deliver('succeeded')
    b.m._tick()
    assert b.m._target_identity_confirmed()
    assert mode_events(b)==['idle']
    assert [e[0] for e in b.events]==['mode','navigate','state']
    assert b.events[1]==('navigate',(1.,2.,0.,'carried_shelf_retreat'),{'profile':'carried_retreat'})
    assert b.m.state=='CLEAR_SHELF_WITH_BOOK' and b.p.mode=='idle'
    assert not b.p.book_history and not b.p.bin_rgb_frames and not b.p.bin_depth_frames
    assert b.m.detected_row==1 and b.m.row_confirmed
    b.m._tick()
    assert mode_events(b)==['idle']


@pytest.mark.parametrize('model',[None,'book_col_3_row_3_red','book_col_3_row_2_blue'])
def test_unconfirmed_target_preserves_original_abort_before_idle_or_navigation(bus,model):
    b=bus;b.m.target_book_model=model;b.deliver('succeeded');b.m._tick()
    assert b.events==[('abort','target_book_identity_not_confirmed')]
    assert b.p.mode=='books'


@pytest.mark.parametrize('payload',[None,('started','pick'),('succeeded','stow')])
def test_no_idle_while_pick_pending_or_other_command_succeeds(bus,payload):
    b=bus
    if payload is not None:b.deliver(*payload)
    b.m._tick()
    assert b.m.state=='PICK' and not b.events and b.p.mode=='books'
    b.p._process()
    assert b.books.call_count==1


@pytest.mark.parametrize('event',['failed','rejected','cancelled'])
def test_failed_pick_retains_original_retry_idle_and_head_sequence(bus,event):
    b=bus;b.deliver(event);b.m._tick()
    assert mode_events(b)==['idle']
    assert [e[0] for e in b.events]==['mode','manipulation','state']
    assert b.events[1][1]['event']=='look_book_row_1'
    assert b.m.state=='RETRY_HEAD_BOOKS' and b.m.book_point is None


@pytest.mark.parametrize('reason',['pick_recovery_failed','pick_payload_lost'])
def test_terminal_pick_failure_does_not_enter_new_idle_success_branch(bus,reason):
    b=bus;b.deliver('failed',reason=reason);b.m._tick()
    assert b.events==[('abort',reason)]


def test_retry_restores_confirmed_row_books_and_requires_a_new_processable_frame(bus):
    b=bus;b.p._process();assert b.books.call_count==1
    b.deliver('failed');b.m._tick()
    b.deliver('succeeded',command='look_book_row_1');b.m._tick()
    assert b.m.state=='REACQUIRE_BOOK' and b.p.mode=='books'
    assert b.p.confirmed_book_row==1 and b.m.book_point is None
    assert mode_events(b)==['idle','books']
    b.p._process()  # Original producer deduplication still rejects the consumed pair.
    assert b.books.call_count==1
    b.clock.ns+=100_000_000
    b.p._on_rgb(image('rgb',b.clock.ns))
    b.p._on_depth(image('depth',b.clock.ns))
    b.p._process()
    assert b.books.call_count==2


def test_successful_carry_stays_idle_until_original_head_bin_success(bus):
    b=bus;b.deliver('succeeded');b.m._tick()
    b.m.nav_event={'event':'reached'};b.m._tick()
    assert b.m.state=='COMPACT_TRANSPORT' and b.p.mode=='idle'
    b.deliver('succeeded',command='compact_transport');b.m._tick()
    assert b.m.state=='RETURN_START' and b.p.mode=='idle'
    b.m.nav_event={'event':'reached'};b.m._tick()
    assert b.m.state=='HEAD_BIN' and b.p.mode=='idle'
    b.m._tick()
    assert mode_events(b)==['idle']
    b.deliver('succeeded',command='look_bin');b.m._tick()
    assert b.m.state=='FIND_BIN' and b.p.mode=='bin' and b.m.bin_point is None
    assert mode_events(b)==['idle','bin']
    assert not b.p.bin_rgb_frames and not b.p.bin_depth_frames


@pytest.mark.parametrize('rgb_ns,depth_ns,detected',[
    (1_000_000_000,1_980_000_000,False),
    (1_980_000_000,1_000_000_000,False),
    (1_000_000_000,1_000_000_000,False),
    (1_980_000_000,1_980_000_000,True),
])
def test_bin_reactivation_preserves_queue_reset_and_original_fresh_pair_admission(bus,rgb_ns,depth_ns,detected):
    b=bus;b.deliver('succeeded');b.m._tick()
    # Cached images are still present in idle, but must not seed the bin queues.
    assert b.p.latest_rgb_message is not None and b.p.latest_depth_message is not None
    b.m.state='HEAD_BIN';b.deliver('succeeded',command='look_bin');b.m._tick()
    b.p._process()
    b.bins.assert_not_called()
    assert not b.p.bin_rgb_frames and not b.p.bin_depth_frames
    b.p._on_rgb(image('rgb',rgb_ns));b.p._on_depth(image('depth',depth_ns))
    b.p._process()
    assert b.bins.call_count==int(detected)
    assert not any(c.args[0]=='bin_verified' for c in b.p._publish_status.call_args_list)


def test_idle_keeps_original_image_conversion_readiness_and_stamps(bus):
    b=bus;b.deliver('succeeded');b.m._tick()
    rgb=image('rgb',b.clock.ns);depth=image('depth',b.clock.ns)
    b.p._on_rgb(rgb);b.p._on_depth(depth);b.p._process()
    assert b.p.latest_rgb_message is rgb and b.p.latest_depth_message is depth
    assert b.p.latest_rgb.shape==(2,2,3) and b.p.latest_depth.shape==(2,2)
    assert b.p._ready() and b.p.last_processed_ns==b.clock.ns
    b.books.assert_not_called();b.bins.assert_not_called()
    assert not b.p.bin_rgb_frames and not b.p.bin_depth_frames


def test_existing_dry_run_identity_bypass_still_reaches_same_retreat(bus):
    b=bus;b.m.dry_run=True;b.m.target_book_model=None;b.deliver('succeeded');b.m._tick()
    assert b.m.state=='CLEAR_SHELF_WITH_BOOK' and mode_events(b)==['idle']
