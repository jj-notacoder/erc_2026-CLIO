"""Private ROS-free geometry process. The parent alone can command the robot."""
import os
from pathlib import Path
import signal
import sys
import traceback

from .geometry_process_protocol import (
    MAX_FRAME_BYTES, MAX_QUERY_BYTES, decode_frame, encode_frame, query_from_wire,
)


def _receive(maximum):
    data = sys.stdin.buffer.readline(maximum + 1)
    if not data:
        return None
    if not data.endswith(b'\n'):
        raise ValueError('truncated geometry frame')
    return decode_frame(data, maximum)


def _send(value):
    sys.stdout.buffer.write(encode_frame(value))
    sys.stdout.buffer.flush()


def main():
    # This entry point is deliberately Linux/POSIX only; normal unsupported
    # callers retain the synchronous backend before constructing a pool.
    if os.name != 'posix' or not sys.platform.startswith('linux'):
        raise RuntimeError('geometry workers require Linux process ownership')
    import ctypes
    # A killed supervisor cannot orphan these separately owned process groups.
    # Avoid preexec_fn in the multithreaded ROS parent; set this in the child.
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise RuntimeError('cannot establish geometry parent-death protection')
    import resource
    initial = _receive(MAX_FRAME_BYTES)
    if type(initial) is not dict or set(initial) != {
            'kind', 'token', 'worker', 'identity', 'epoch', 'scene', 'scene_id',
            'address_space_bytes', 'cpu_seconds', 'geometry_id', 'parent_pid'} or initial['kind'] != 'begin':
        raise ValueError('invalid geometry bootstrap')
    if type(initial['parent_pid']) is not int or initial['parent_pid']<=1 or os.getppid()!=initial['parent_pid']:
        raise RuntimeError('geometry worker parent changed before bootstrap')
    if type(initial['token']) is not str or len(initial['token']) != 64:
        raise ValueError('invalid geometry session token')
    if type(initial['worker']) is not int or not 0 <= initial['worker'] < 8:
        raise ValueError('invalid geometry worker number')
    memory, cpu = initial['address_space_bytes'], initial['cpu_seconds']
    if type(memory) is not int or not 1024**3 <= memory <= 4*1024**3:
        raise ValueError('invalid worker address space bound')
    if type(cpu) is not int or not 1 <= cpu <= 180:
        raise ValueError('invalid worker CPU bound')
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Imports and ordinary installed-resource identity are verified inside the
    # child. No live node, audit path, serialized mesh, or ROS object is sent.
    from .installed_geometry_identity import InstalledGeometryIdentity
    from .pure_geometry_owner import RobotGeometryOwner
    identity = InstalledGeometryIdentity.from_descriptor(initial['identity'])
    if Path(__file__).resolve() != identity.module_files.get('erc_phase1_solution.geometry_process_worker'):
        raise RuntimeError('geometry worker entry-point origin mismatch')
    from .pickup_geometry_owner import PickupGeometryOwner, SCENE_KIND, POST_RETREAT_SCENE_KIND
    from .empty_pickup_geometry_owner import EmptyPickupGeometryOwner, SCENE_KIND as EMPTY_SCENE_KIND
    pickup = initial['scene'].get('kind') in (SCENE_KIND, POST_RETREAT_SCENE_KIND)
    empty = initial['scene'].get('kind') == EMPTY_SCENE_KIND
    owner = (EmptyPickupGeometryOwner(identity) if empty else
             PickupGeometryOwner(identity) if pickup else RobotGeometryOwner(identity))
    scene_id = owner.begin_scene(epoch=initial['epoch'], scene=initial['scene'])
    if scene_id != initial['scene_id']:
        raise RuntimeError('worker scene identity mismatch')
    from .geometry_process_models import scene_model_signature
    geometry_id = owner.model_signature() if pickup or empty else scene_model_signature(owner._scene)
    if geometry_id != initial['geometry_id']:
        raise RuntimeError('parent/worker actual geometry model mismatch')
    identity.verify()
    binding = dict(token=initial['token'], worker=initial['worker'], pid=os.getpid(),
        source_id=identity.source_id, model_id=identity.model_id,
        epoch=initial['epoch'], scene_id=scene_id, geometry_id=geometry_id)
    _send(dict(kind='ready', **binding))
    while True:
        message = _receive(MAX_QUERY_BYTES)
        if message is None:
            return
        if (set(message)=={'kind','token','last_request_id'} and message['kind']=='finish'
                and message['token']==initial['token']):
            if type(message['last_request_id']) is not int or message['last_request_id']!=owner._last_request_id:
                raise ValueError('geometry completion request boundary mismatch')
            identity.verify()
            if (owner.model_signature() if pickup or empty else scene_model_signature(owner._scene))!=geometry_id:
                raise RuntimeError('worker geometry changed during evaluation')
            _send(dict(kind='finished',**binding,last_request_id=owner._last_request_id))
            # Remain alive until parent-owned closure. EOF cannot race the
            # parent's completion acknowledgement with a false death signal.
            if _receive(MAX_QUERY_BYTES) is not None:
                raise ValueError('geometry message after completion boundary')
            return
        if set(message) != {'kind', 'token', 'query'} or message['kind'] != 'query' or message['token'] != initial['token']:
            raise ValueError('invalid geometry query envelope')
        query = query_from_wire(message['query'])
        delta = owner.evaluate_independent(query)
        _send(dict(kind='result', token=initial['token'], worker=initial['worker'],
                   pid=os.getpid(), delta=delta.to_wire()))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # stderr is bounded/drained by the parent. An absent or incomplete
        # reply cannot be interpreted as a collision-free sample.
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
