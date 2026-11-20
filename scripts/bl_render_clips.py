import bpy, sys, math, json, mathutils, os
sys.path.insert(0, '/home/user/work/scripts')
from bl_scene import *
from clips import *
from bpy_extras.object_utils import world_to_camera_view
names = sys.argv[sys.argv.index('--') + 1:]
s, groups, c = build()
C = {g: centroid(g, groups) for g in groups}
cen = {g: C[g][0] for g in groups}
cen['cage'] = (cen['sternum'] + cen['vertebrae']) / 2; cen['cage'].z = cen['sternum'].z - 40
cen['forearm'] = (cen['radius'] + cen['ulna']) / 2
handv = C['carpals'][1] + C['metacarpals'][1] + C['phalanges'][1]
cen['hand'] = sum(handv, mathutils.Vector()) / len(handv)
cen['upper'] = mathutils.Vector((-95, -100, 1070))
# humerus feature points
hv = C['humerus'][1]
def band(z0, z1): return [v for v in hv if z0 <= v.z <= z1]
zmin = min(v.z for v in hv); zmax = max(v.z for v in hv); Lh = zmax - zmin
def pick(zf0, zf1, score):
    b = band(zmin + zf0 * Lh, zmin + zf1 * Lh); return max(b, key=score)
b = band(zmin + .83 * Lh, zmin + .88 * Lh); xc = sorted(v.x for v in b)[len(b) // 2]
F = {}
F['bicipital'] = max([v for v in b if abs(v.x - (xc - 6)) < 6], key=lambda v: -v.y)
b2 = band(zmin + .08 * Lh, zmin + .14 * Lh); xc2 = sorted(v.x for v in b2)[len(b2) // 2]
F['coronoid'] = max([v for v in b2 if abs(v.x - xc2) < 6], key=lambda v: -v.y)
F['olecranon'] = max([v for v in b2 if abs(v.x - xc2) < 6], key=lambda v: v.y)
b3 = band(zmin + .45 * Lh, zmin + .6 * Lh); yc3 = sum(v.y for v in b3) / len(b3); xc3 = sum(v.x for v in b3) / len(b3)
F['radialg'] = max(b3, key=lambda v: (v.y - yc3) + 0.6 * (xc3 - v.x))
P = dict(cen); P.update(F)
rv = [v for v in C['ribs'][1] if abs(v.z - (cen['sternum'].z - 60)) < 25 and v.x < 0]
if rv: P['ribs'] = min(rv, key=lambda v: v.x + 0.5 * v.y)
cv_ = [v for v in C['cartilage'][1] if v.x < -30]
if cv_: P['cartilage'] = sum(cv_, mathutils.Vector()) / len(cv_)
P['shoulder'] = max(hv, key=lambda v: v.z); P['fingertip'] = min(C['phalanges'][1], key=lambda v: v.z)
P['elbow'] = min(hv, key=lambda v: v.z); P['midline_top'] = mathutils.Vector((0, cen['sternum'].y, cen['sternum'].z + 160))
P['midline_bot'] = mathutils.Vector((0, cen['sternum'].y, cen['sternum'].z - 330))
P['front'] = cen['sternum'] + mathutils.Vector((0, -60, 0)); P['back'] = cen['vertebrae'] + mathutils.Vector((0, 70, 0))
def sm(x): x = max(0, min(1, x)); return x * x * (3 - 2 * x)
def interp_cam(keys, t):
    if t <= keys[0][0]: k0 = k1 = keys[0]; a = 0
    elif t >= keys[-1][0]: k0 = k1 = keys[-1]; a = 0
    else:
        for i in range(len(keys) - 1):
            if keys[i][0] <= t <= keys[i + 1][0]: k0, k1 = keys[i], keys[i + 1]; a = sm((t - k0[0]) / (k1[0] - k0[0])); break
    tg = lambda k: P[k[1]] if isinstance(k[1], str) else mathutils.Vector(k[1])
    tgt = tg(k0).lerp(tg(k1), a); d = k0[2] + (k1[2] - k0[2]) * a
    az = math.radians(k0[3] + (k1[3] - k0[3]) * a); el = math.radians(k0[4] + (k1[4] - k0[4]) * a)
    return tgt, d, az, el
def statecol(sv):
    return {'hi': (GOLD, 1.0, 0.25), 'on': (IVORY, 1.0, 0), 'gh': (IVORY, 0.28, 0), 'off': (IVORY, 0.0, 0),
            'teal': (TEAL, 1.0, 0.25), 'coral': (CORAL, 1.0, 0.25)}[sv]
def mat_state(hl, t, g):
    cur = None; prev = None; t_on = 0
    for (tk, stt) in hl:
        if tk <= t: prev = cur; cur = stt[g]; t_on = tk
    if cur is None: cur = hl[0][1][g]
    a = sm((t - t_on) / 0.5) if prev is not None else 1
    c1, a1, e1 = statecol(cur)
    if prev is None or prev == cur: return c1, a1, e1
    c0, a0, e0 = statecol(prev)
    return tuple(c0[i] + (c1[i] - c0[i]) * a for i in range(3)), a0 + (a1 - a0) * a, e0 + (e1 - e0) * a
for name in names:
    cl = CLIPS[name]; out = f'/home/user/work/b3d/{name}'; os.makedirs(out, exist_ok=True)
    t0, t1 = cl['t']; n = int(round((t1 - t0) * FPS3D)) + 1; proj = []
    for f in range(n):
        t = t0 + f / FPS3D
        tgt, d, az, el = interp_cam(cl['cam'], t)
        c.location = tgt + mathutils.Vector((d * math.sin(az) * math.cos(el), -d * math.cos(az) * math.cos(el), d * math.sin(el)))
        c.rotation_euler = (tgt - c.location).to_track_quat('-Z', 'Y').to_euler()
        for g, dd in groups.items():
            col, al, em = mat_state(cl['hl'], t, g)
            dd['P'].inputs['Base Color'].default_value = (*col, 1)
            if 'Emission Color' in dd['P'].inputs:
                dd['P'].inputs['Emission Color'].default_value = (*col, 1); dd['P'].inputs['Emission Strength'].default_value = em
            dd['M'].inputs[0].default_value = al
            for o in dd['objs']: o.hide_render = al < 0.01
        bpy.context.view_layer.update()
        pr = {}
        for k, v in P.items():
            q = world_to_camera_view(s, c, v); pr[k] = (q.x, 1 - q.y, q.z)
        proj.append(dict(t=t, az=math.degrees(az), p=pr))
        fp = f'{out}/{f:05d}.png'
        if not os.path.exists(fp):
            s.render.filepath = fp; bpy.ops.render.render(write_still=True)
        print('FRAME', name, f, n, flush=True)
    json.dump(proj, open(f'{out}/proj.json', 'w'))
print('ALLCLIPS_DONE')
