"""Stock-bin registration from visible RGB-D surface planes.

No ROS, filesystem, evaluator poses, world location, ray-heading or centered-table
prior. BGR matches the production CvBridge contract; depth is optical Z in metres.
The caller authenticates the pinned mesh and binds one depth/RGB/TF epoch.
"""
from __future__ import annotations
import itertools
import math
import numpy as np

BIN_MESH_SHA256 = '7c036fe096a50eadc8c32f30837964c3bbbb012cf7a45b874eb7a6f96973aa2a'
MODEL = 'stock_bin_visible_inner_planes_v1'
OUTER = np.asarray([[-.155, -.105, -.305], [.155, .105, .255]])
CAVITY = np.asarray([[-.145, -.095, -.245], [.145, .105, .245]])

def _invalid(reason, **quality):
    return dict(valid=False, model=MODEL, frame='base_footprint', reason=reason,
                quality=quality)

def _array(value, shape):
    a=np.asarray(value,dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError('invalid finite input shape')
    return a

def _planes(points):
    rng=np.random.default_rng(736)
    remain=np.arange(len(points)); output=[]
    for _ in range(7):
        if len(remain)<80:break
        p=points[remain]
        if len(p)>4000:p=p[np.linspace(0,len(p)-1,4000,dtype=int)]
        best=None
        for _ in range(160):
            a,b,c=p[rng.choice(len(p),3,replace=False)]
            n=np.cross(b-a,c-a);length=np.linalg.norm(n)
            if length<1e-6:continue
            n/=length; count=int(np.count_nonzero(np.abs((p-a)@n)<.0015))
            if best is None or count>best[0]:best=(count,n,a)
        if best is None:break
        _,n,a=best;q=points[remain];mask=np.abs((q-a)@n)<.0015
        if np.count_nonzero(mask)<80:break
        for _ in range(2):
            inside=q[mask];center=inside.mean(0)
            _,sv,vh=np.linalg.svd(inside-center,full_matrices=False)
            n=vh[-1];mask=np.abs((q-center)@n)<.0015
        ids=remain[mask]
        if len(ids)<80:break
        n*=1 if n[np.argmax(np.abs(n))]>=0 else -1
        inside=points[ids];dist=(inside-center)@n
        output.append(dict(n=n,center=center,points=inside,count=len(ids),
                           rms=float(np.sqrt(np.mean(dist**2)))))
        remain=remain[~mask]
    return output

def _span(p, axis):
    return float(np.quantile(p@axis,.98)-np.quantile(p@axis,.02))

def _surface_residual(local):
    """Plane/AABB feature-consistency residual, not collision/mesh distance.

    Coordinates derive from the pinned stock CAD planes. Finite supports avoid
    accepting a red object solely because it shares an extended infinite plane.
    Sloped handle rectangles are conservative association regions only.
    """
    surfaces=[]
    def face(n,d,lo,hi):surfaces.append((np.asarray(n),d,np.asarray(lo),np.asarray(hi)))
    face([0,1,0],-.095,[-.145,-.095,-.259645],[.145,-.095,.245])
    face([0,1,0],-.105,[-.155,-.105,-.269645],[.155,-.105,.255])
    for sign in (-1,1):
        for x in (.145,.155):
            face([1,0,0],sign*x,[sign*x,-.105,-.305],[sign*x,.105,.255])
    for z in (.245,.255):face([0,0,1],z,[-.155,-.105,z],[.155,.105,z])
    for sign in (-1,1):
        xs=sorted([sign*.145,sign*.155])
        face([0,1,0],.105,[xs[0],.105,-.255],[xs[1],.105,.255])
    face([0,1,0],.105,[-.155,.105,.245],[.155,.105,.255])
    face([0,1,0],.105,[-.155,.105,-.255],[.155,.105,-.245])
    for d in (-.275760176,-.265760176):
        face([0,-.372302953,.928111260],d,[-.155,-.022,-.305],[.155,.105,-.245])
    for z in (-.305,-.295):face([0,0,1],z,[-.155,-.055,z],[.155,-.019,z])
    for d in (-.254307145,-.244307186):
        face([0,.70710678,.70710678],d,[-.155,-.095,-.305],[.155,-.05,-.259])
    best=np.full(len(local),np.inf)
    for n,d,lo,hi in surfaces:
        outside=np.maximum(np.maximum(lo-local,local-hi),0).max(1)
        best=np.minimum(best,np.maximum(np.abs(local@n-d),outside))
    return best

def fit_bin_points(points, camera_origin, *, table_scene=None):
    """Fit only if floor, paired inner sides and rear inner wall are observable.

    The two side offsets locate the lateral center. The tall perpendicular
    inner rear wall fixes the long-axis center at its known CAD offset. A
    partial cloud's minimum/maximum never supplies an unseen wall or center.
    """
    points=np.asarray(points,dtype=float)
    camera=_array(camera_origin,(3,))
    if points.ndim!=2 or points.shape[1]!=3 or not np.all(np.isfinite(points)):
        return _invalid('invalid_points')
    if len(points)<500:return _invalid('insufficient_red_depth')
    if len(points)>20000:points=points[np.linspace(0,len(points)-1,20000,dtype=int)]
    groups=_planes(points)
    floors=[i for i,g in enumerate(groups) if abs(g['n'][2])>math.cos(math.radians(3)) and g['count']>=300]
    walls=[i for i,g in enumerate(groups) if abs(g['n'][2])<math.sin(math.radians(3)) and g['count']>=100]
    candidates=[];rejects={}
    def rejected(name):rejects[name]=rejects.get(name,0)+1
    for fi in floors:
      floor=groups[fi];up=floor['n'].copy()
      if up[2]<0:up=-up
      for ai,bi in itertools.combinations(walls,2):
        a,b=groups[ai],groups[bi]
        dot=float(a['n']@b['n'])
        if abs(dot)<math.cos(math.radians(2)):continue
        side=a['n']+b['n']*np.sign(dot);side-=up*(side@up);side/=np.linalg.norm(side)
        width=abs(float(np.median(a['points']@side)-np.median(b['points']@side)))
        if abs(width-.29)>.008:rejected('wrong_inner_side_separation');continue
        for ri in walls:
          if ri in (ai,bi):continue
          rear=groups[ri]
          if abs(float(rear['n']@side))>math.sin(math.radians(2)):continue
          long=np.cross(side,up)
          if long@(rear['center']-floor['center'])<0:long=-long;side=-side
          if (long@(camera-rear['center']))*(long@(floor['center']-rear['center']))<=0:
              rejected('rear_not_visible_from_cavity_side');continue
          if (_span(rear['points'],side)<.18 or _span(rear['points'],up)<.10
              or any(_span(g['points'],long)<.15 or _span(g['points'],up)<.08 for g in (a,b))
              or _span(floor['points'],side)<.10 or _span(floor['points'],long)<.12):
              rejected('insufficient_two_dimensional_plane_support');continue
          x=.5*(np.median(a['points']@side)+np.median(b['points']@side))
          y=np.median(floor['points']@up)
          z=np.median(rear['points']@long)-.245
          rotation=np.column_stack((side,up,long))
          center=rotation@np.asarray([x,y,z])
          origin=center+.095*up
          local=(points-origin)@rotation
          residual=_surface_residual(local)
          coverage=float(np.mean(residual<=.003))
          residual_p95=float(np.quantile(residual,.95))
          if coverage<.95 or residual_p95>.003:rejected('visible_points_disagree_with_finite_stock_surfaces');continue
          # Rear-bottom continuity and the 200 mm floor/rim separation are
          # physical feature associations, not assumptions about occluded edges.
          rear_y=(rear['points']-center)@up
          if np.quantile(rear_y,.02)>.025 or abs(np.quantile(rear_y,.98)-.2)>.025:
              rejected('rear_floor_rim_registration_not_observed');continue
          if table_scene is not None:
              try:
                  tr=_array(table_scene['rotation'],(3,3));to=_array(table_scene['origin'],(3,))
                  if (table_scene['valid'] is not True or table_scene['frame']!='base_footprint'
                      or table_scene['model']!='erc_table_two_edge_rgbd_v1'
                      or not np.allclose(tr.T@tr,np.eye(3),atol=1e-6)
                      or abs(np.linalg.det(tr)-1)>1e-6 or tr[2,2]<math.cos(math.radians(3))):
                      raise ValueError()
                  # TableSceneObstacle's origin is the measured tabletop center.
                  above=float((center-to)@tr[:,2])
                  if abs(above-.010)>.015:raise ValueError()
              except (KeyError,ValueError,TypeError):
                  rejected('inconsistent_optional_table_height');continue
          candidates.append((center,origin,rotation,dict(
            floor_points=floor['count'],side_points=[a['count'],b['count']],rear_points=rear['count'],
            observed_inner_width_m=width,finite_cad_coverage=coverage,
            finite_surface_residual_p95_m=residual_p95,
            plane_rms_m=[floor['rms'],a['rms'],b['rms'],rear['rms']],
            floor_rear_registration_observed=True,
            feature_indices=[fi,ai,bi,ri],partial_cloud_bounds_used_for_center=False)))
    if not candidates:return _invalid('no_identifiable_inner_plane_registration',plane_count=len(groups),rejections=rejects)
    best=min(candidates,key=lambda c:c[3]['finite_surface_residual_p95_m'])
    for c in candidates:
        angle=math.acos(float(np.clip((np.trace(best[2].T@c[2])-1)/2,-1,1)))
        if np.linalg.norm(c[0]-best[0])>.005 or angle>math.radians(1):
            return _invalid('ambiguous_cad_registration',candidate_count=len(candidates))
    center,origin,r,quality=best
    # Engineering assumptions cover calibration/feature registration; tiny
    # rendered-plane residuals are not used as camera accuracy certificates.
    translation=.005+quality['finite_surface_residual_p95_m']
    rotation_error=.01
    radius=float(np.max(np.linalg.norm(np.asarray(list(itertools.product(*zip(*OUTER)))),axis=1)))
    registration=translation+2*radius*math.sin(rotation_error/2)
    quality.update(accepted_candidates=len(candidates),plane_count=len(groups),rejections=rejects,
                   yaw_long_axis_rad=float(math.atan2(r[1,2],r[0,2])),
                   rotation_condition=float(np.linalg.cond(r)),uncertainty_is_engineering_assumption=True)
    return dict(valid=True,model=MODEL,frame='base_footprint',mesh_sha256=BIN_MESH_SHA256,
       origin=origin.tolist(),rotation=r.tolist(),floor_center=center.tolist(),
       floor_point_model=[0.,-.095,0.],outer_bounds=OUTER.tolist(),cavity_bounds=CAVITY.tolist(),
       modeled_margin_m=.005,registration_margin_m=registration,
       translation_error_m=translation,rotation_error_rad=rotation_error,quality=quality,
       producer_stamp_bound_by_caller=True,uses_evaluator_pose=False,
       nominal_mimic_or_scene_motion_proof=False)

def fit_bin_scene(bgr, depth, intrinsics, camera_to_base, detection, table_scene=None):
    """JSON-safe selected-profile fit; caller supplies its accepted detection.

    Requires aligned optical-Z depth and depth-stamped rigid TF. This function
    owns no clock/identity, so a valid result alone is never live admission.
    """
    try:
        image=np.asarray(bgr);d=np.asarray(depth,dtype=float)
        k=_array(np.asarray(intrinsics).reshape(3,3),(3,3));tf=_array(camera_to_base,(4,4))
        if image.shape!=d.shape+(3,) or d.ndim!=2 or image.dtype!=np.uint8:
            return _invalid('unaligned_bgr_depth')
        if min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1]):raise ValueError()
        if (not np.allclose(tf[3],[0,0,0,1]) or not np.allclose(tf[:3,:3].T@tf[:3,:3],np.eye(3),atol=1e-6)
              or abs(np.linalg.det(tf[:3,:3])-1)>1e-6):raise ValueError()
        bbox=detection.bbox if hasattr(detection,'bbox') else detection['bbox']
        if len(bbox)!=4 or any(isinstance(v,bool) or int(v)!=v for v in bbox):raise ValueError()
        x,y,w,h=map(int,bbox)
        if min(w,h)<10 or min(x,y)<2 or x+w>=d.shape[1]-2 or y+h>=d.shape[0]-2:
            return _invalid('clipped_or_invalid_detection')
    except (ValueError,TypeError,KeyError,AttributeError):return _invalid('invalid_calibration_or_detection')
    bb,gg,rr=image.astype(float).transpose(2,0,1)
    red=(rr>=45)&(rr>1.45*gg)&(rr>1.45*bb)
    selected=np.zeros(d.shape,bool);selected[y:y+h,x:x+w]=True
    v,u=np.nonzero(selected&red&np.isfinite(d)&(d>.2)&(d<4))
    z=d[v,u]
    camera=np.column_stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z))
    points=camera@tf[:3,:3].T+tf[:3,3]
    result=fit_bin_points(points,tf[:3,3],table_scene=table_scene)
    result.setdefault('quality',{}).update(red_depth_points=len(points),bbox=[x,y,w,h],
      input_encoding='BGR8 + optical-Z metres',red_filter='R>=45 and R>1.45*G and R>1.45*B',
      table_used_for_horizontal_registration=False)
    return result
