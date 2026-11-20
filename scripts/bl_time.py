import bpy, sys, math, mathutils, time
sys.path.insert(0,'/home/user/work/scripts'); from bl_scene import *
s, groups, c = build()
for g,d in groups.items():
    d['P'].inputs['Base Color'].default_value = (*(GOLD if g=='clavicle' else IVORY),1); d['M'].inputs[0].default_value = 1 if g=='clavicle' else 0.35
tgt = mathutils.Vector((-80, -110, 1150))
for k in range(3):
    az, el, dist = math.radians(-20+k*3), math.radians(8), 1500
    c.location = tgt + mathutils.Vector((dist*math.sin(az)*math.cos(el), -dist*math.cos(az)*math.cos(el), dist*math.sin(el)))
    c.rotation_euler = (tgt - c.location).to_track_quat('-Z','Y').to_euler()
    t=time.time(); s.render.filepath = f'/home/user/work/b3d/tt{k}.png'; bpy.ops.render.render(write_still=True); print('RT', round(time.time()-t,2))
