"""Bounded exact query/reply protocol and ordered committed geometry state."""
from dataclasses import dataclass
import copy
import json
import math

MAX_FRAME_BYTES = 512 * 1024
MAX_QUERY_BYTES = 8192


def encode_frame(value, maximum=MAX_FRAME_BYTES):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
    if len(data) > maximum:
        raise ValueError('geometry message exceeds byte bound')
    return data


def decode_frame(data, maximum=MAX_FRAME_BYTES):
    if type(data) is not bytes or len(data) > maximum or not data.endswith(b'\n') or b'\n' in data[:-1]:
        raise ValueError('invalid geometry message frame')
    def invalid(value):
        raise ValueError('nonfinite geometry wire value: '+value)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate geometry message field')
            value[key] = item
        return value
    value = json.loads(data, parse_constant=invalid, object_pairs_hook=unique)
    if type(value) is not dict:
        raise ValueError('geometry message must be an object')
    return value


def query_to_wire(query):
    from .empty_pickup_geometry_owner import EmptyPickupGeometryQuery
    if type(query) is EmptyPickupGeometryQuery:
        query.validated()
        return dict(empty_pickup_operation=query.operation,
                    sample=query_to_wire(query.sample), input_sha256=query.input_sha256)
    from .place_volume_geometry import PlaceVolumeQuery
    if type(query) is PlaceVolumeQuery:
        query.validated()
        return dict(place_volume_operation=query.operation,
                    sample=query_to_wire(query.sample), input_sha256=query.input_sha256)
    from .pickup_geometry_owner import PickupGeometryQuery
    if type(query) is PickupGeometryQuery:
        query.validated()
        return dict(pickup_operation=query.operation,
                    sample=query_to_wire(query.sample),
                    input_sha256=query.input_sha256)
    query.validated()
    return dict(request_id=query.request_id, epoch=query.epoch,
        source_id=query.source_id, model_id=query.model_id, scene_id=query.scene_id,
        q=query.q.hex(), right=query.right.hex(), head=query.head.hex(),
        aperture=query.aperture, loaded=query.loaded, input_sha256=query.input_sha256)


def query_from_wire(value):
    from .pure_geometry_owner import GeometryQuery
    if type(value) is dict and 'empty_pickup_operation' in value:
        from .empty_pickup_geometry_owner import EmptyPickupGeometryQuery
        if set(value) != {'empty_pickup_operation', 'sample', 'input_sha256'}:
            raise ValueError('empty pickup query fields mismatch')
        encode_frame(value, MAX_QUERY_BYTES)
        sample = query_from_wire(value['sample'])
        if type(sample) is not GeometryQuery:
            raise ValueError('nested empty pickup queries are invalid')
        query = EmptyPickupGeometryQuery(sample, value['empty_pickup_operation']).validated()
        if query.input_sha256 != value['input_sha256']:
            raise ValueError('empty pickup query hash mismatch')
        return query
    if type(value) is dict and 'place_volume_operation' in value:
        from .place_volume_geometry import PlaceVolumeQuery
        if set(value) != {'place_volume_operation', 'sample', 'input_sha256'}:
            raise ValueError('PLACE volume query fields mismatch')
        encode_frame(value, MAX_QUERY_BYTES)
        sample=query_from_wire(value['sample'])
        if type(sample) is not GeometryQuery:
            raise ValueError('nested PLACE volume queries are invalid')
        query=PlaceVolumeQuery(sample,value['place_volume_operation']).validated()
        if query.input_sha256 != value['input_sha256']:
            raise ValueError('PLACE volume query hash mismatch')
        return query
    if type(value) is dict and 'pickup_operation' in value:
        from .pickup_geometry_owner import PickupGeometryQuery
        if set(value) != {'pickup_operation', 'sample', 'input_sha256'}:
            raise ValueError('pickup query fields mismatch')
        encode_frame(value, MAX_QUERY_BYTES)
        sample = query_from_wire(value['sample'])
        if type(sample) is not GeometryQuery:
            raise ValueError('nested pickup queries are invalid')
        query = PickupGeometryQuery(sample, value['pickup_operation']).validated()
        if query.input_sha256 != value['input_sha256']:
            raise ValueError('pickup query hash mismatch')
        return query
    if type(value) is not dict or set(value) != {
        'request_id','epoch','source_id','model_id','scene_id','q','right','head',
        'aperture','loaded','input_sha256'
    }:
        raise ValueError('geometry query fields mismatch')
    encode_frame(value, MAX_QUERY_BYTES)
    current = dict(value)
    expected = current.pop('input_sha256')
    for field, size in (('q',128),('right',112),('head',32)):
        if type(current[field]) is not str or len(current[field]) != size:
            raise ValueError('geometry vector encoding length mismatch')
        current[field] = bytes.fromhex(current[field])
    query = GeometryQuery(**current).validated()
    if query.input_sha256 != expected:
        raise ValueError('geometry query hash mismatch')
    return query


@dataclass(frozen=True)
class SampleDelta:
    """Only this query's state changes; no speculative-prefix accumulation."""
    request_id: int
    epoch: int
    source_id: str
    model_id: str
    scene_id: str
    input_sha256: str
    verdict: bool
    minimum_update: object
    rejection_update: object
    table_touched: bool
    table_intersection: object

    def to_wire(self):
        value = dict(self.__dict__)
        self.validate()
        encode_frame(value, MAX_QUERY_BYTES)
        return value

    def validate(self):
        if type(self.request_id) is not int or self.request_id < 0 or type(self.epoch) is not int or self.epoch < 0:
            raise ValueError('invalid geometry result identity')
        if any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
               for v in (self.source_id,self.model_id,self.scene_id,self.input_sha256)):
            raise ValueError('invalid geometry result digest')
        if type(self.verdict) is not bool or type(self.table_touched) is not bool:
            raise ValueError('geometry result flags must be boolean')
        if self.minimum_update is not None and (type(self.minimum_update) not in (int,float)
                                                or not math.isfinite(self.minimum_update)):
            raise ValueError('invalid minimum update')
        if self.rejection_update is not None and (type(self.rejection_update) is not dict
                or type(self.rejection_update.get('reason')) is not str):
            raise ValueError('invalid rejection update')
        if not self.verdict and self.rejection_update is None:
            raise ValueError('rejected result has no reason')
        if self.verdict and self.rejection_update is not None:
            raise ValueError('successful result carries a rejection')
        if self.table_intersection is not None and type(self.table_intersection) is not str:
            raise ValueError('invalid table intersection')
        if not self.table_touched and self.table_intersection is not None:
            raise ValueError('untouched table has an update')
        encode_frame(dict(self.__dict__), MAX_QUERY_BYTES)
        return self

    def matches(self, query):
        self.validate()
        return (self.request_id,self.epoch,self.source_id,self.model_id,self.scene_id,self.input_sha256)==(
            query.request_id,query.epoch,query.source_id,query.model_id,query.scene_id,query.input_sha256)

    @classmethod
    def from_wire(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('geometry result fields mismatch')
        encode_frame(value, MAX_QUERY_BYTES)
        return cls(**value).validate()


def sample_key(query):
    return query.q, query.aperture, query.loaded, query.right, query.head


def commit_sample(scene, query, delta, *, cancelled):
    """Commit one caller-ordered sample; no suffix result may call this."""
    if cancelled():
        return scene._reject('cancelled')
    query.validated()
    key = sample_key(query)
    if key in scene.cache:
        scene.samples += 1
        scene.cache_hits += 1
        return True
    if type(delta) is not SampleDelta or not delta.matches(query):
        raise RuntimeError('geometry result does not match committed sample')
    scene.samples += 1
    if delta.minimum_update is not None:
        scene.minimum_moving_left_z = min(scene.minimum_moving_left_z, delta.minimum_update)
    if delta.rejection_update is not None:
        scene.last_rejection = copy.deepcopy(delta.rejection_update)
    if delta.table_touched and scene.table is not None:
        scene.table.last_intersection = delta.table_intersection
    if delta.verdict:
        scene.cache[key] = True
    return delta.verdict
