"""Deferred actual ROS message tests for selected stock position; no simulator."""
import pytest
pytest.importorskip('rclpy')
from erc_phase1_solution import stock_gripper_close as stock
from test_stock_gripper_close import close_fixture

@pytest.mark.parametrize('target',[0.,.017,.069])
def test_selected_target_is_the_single_real_ros_command_and_result(monkeypatch,target):
    controller,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    n.stock_gripper_close_target_position=target
    n._stock_close_controller=controller=stock.StockGripperClose(n)
    assert controller.run()[0]
    assert len(sent)==1
    assert list(sent[0].points[0].positions)==[target]
    assert sent[0].points[0].time_from_start.sec==1
    assert sent[0].points[0].time_from_start.nanosec==0
    assert events[-1][1]['commanded_position']==target
    assert next(fields for event,fields in events if event=='gripper_close_diagnostic')['target']==target
    assert not n._adaptive_hold_sent

@pytest.mark.parametrize('target',[-1e-12,.069000000001,float('nan'),float('inf')])
def test_invalid_target_never_reaches_public_writer(monkeypatch,target):
    _,n,_,_,sent,_,_=close_fixture(monkeypatch)
    n.stock_gripper_close_target_position=target
    with pytest.raises(ValueError,match='public range'):stock.StockGripperClose(n)
    assert sent==[]

def test_selected_target_is_not_overwritten_when_cancel_replaces_command(monkeypatch):
    _,n,now,wall,sent,events,sensors=close_fixture(monkeypatch)
    n.stock_gripper_close_target_position=.017
    n._stock_close_controller=controller=stock.StockGripperClose(n)
    began=now.nanoseconds
    def cancel():
        sensors()
        if now.nanoseconds-began>=20_000_000:n._cancel.set()
    wall.hook=cancel
    assert not controller.run()[0]
    assert list(sent[0].points[0].positions)==[.017]
    assert len(sent)==2 and n._adaptive_hold_sent
    assert list(sent[1].points[0].positions)==pytest.approx([n.joints['gripper_left_finger_joint']])
