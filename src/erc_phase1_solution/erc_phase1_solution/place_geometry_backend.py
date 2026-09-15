"""Explicit node selection of the managed, portable PLACE geometry backend."""
import time

from .geometry_process_pool import checked_geometry_process_workers


def checked_place_parallel_geometry_enabled(value):
    if type(value) is not bool:
        raise ValueError('place_parallel_geometry_enabled must be a Boolean')
    return value


def checked_place_carried_volume_parallel_enabled(value):
    if type(value) is not bool:
        raise ValueError('place_carried_volume_parallel_enabled must be a Boolean')
    return value


def initialize_place_geometry_backend(node,resolve_package):
    """Resolve normal installed identity once, only for an enabled live node."""
    enabled=checked_place_parallel_geometry_enabled(getattr(node,'place_parallel_geometry_enabled',False))
    node._place_parallel_geometry_identity=None
    node._place_parallel_geometry_epoch=0
    node._place_parallel_geometry_active=False
    if enabled:
        from .installed_geometry_identity import InstalledGeometryIdentity,required_package_names
        description=resolve_package('erc_description')
        shares={name:resolve_package(name) for name in required_package_names(description)}
        node._place_parallel_geometry_identity=InstalledGeometryIdentity.capture(package_shares=shares)


def plan_node_scene_checked_place(node,local_planner,*args,**kwargs):
    """Return a plan only after worker closure and final parent admissions.

    The default path receives exactly the original planner arguments. Selected
    calls reserve a unique epoch even when planning fails. Concurrent calls are
    rejected rather than sharing a geometry owner or completion boundary.
    """
    if not checked_place_parallel_geometry_enabled(getattr(node,'place_parallel_geometry_enabled',False)):
        return local_planner(node,*args,**kwargs)
    from .geometry_process_parent import ProcessPlanningBackend
    volume_enabled=checked_place_carried_volume_parallel_enabled(
        getattr(node,'place_carried_volume_parallel_enabled',False))
    count = checked_geometry_process_workers(getattr(node, 'geometry_process_workers', 4))
    pool_limits = {} if count == 4 else {'worker_count': count}
    if volume_enabled:
        pool_limits['include_carried_volume']=True
    with node._lock:
        identity=node._place_parallel_geometry_identity
        epoch=node._place_parallel_geometry_epoch
        if (identity is None or type(epoch) is not int or not 0<=epoch<2**63-1
                or node._place_parallel_geometry_active):
            raise RuntimeError('parallel PLACE geometry is unavailable or already active')
        epoch+=1
        node._place_parallel_geometry_epoch=epoch
        node._place_parallel_geometry_active=True
    started=time.monotonic();backend=None;passed=False;failure=None
    try:
        with ProcessPlanningBackend(node,identity,epoch=epoch,**pool_limits) as backend:
            result=local_planner(node,*args,geometry_backend=backend,**kwargs)
        passed=True
        return result
    except BaseException as error:
        failure=error
        raise
    finally:
        with node._lock:
            node._place_parallel_geometry_active=False
        # Only post-exit diagnostics use the existing status stream. Failed
        # telemetry cannot return a failed plan or bypass any motion admission.
        try:
            pool=None if backend is None else backend.pool
            status=(pool.worker_status() if pool is not None else
                    getattr(failure,'geometry_worker_status',None))
            if status is None:
                status=dict(workers_created=0,workers=[],workers_reaped=0,
                    children_closed=True,pool_startup_wall_seconds=None,
                    sequence_wall_seconds=None,query_statistics=None)
            node._publish_status('place_geometry_process_completed',command='place',
                epoch=epoch,passed=passed,wall_seconds=time.monotonic()-started,
                fallback_reason=None if backend is None else backend.fallback_reason,
                failure_type=None if failure is None else type(failure).__name__,
                **status)
        except Exception:
            pass
