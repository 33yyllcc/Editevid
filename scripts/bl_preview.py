import bpy, sys, math, mathutils
sys.path.insert(0,'/home/user/work/scripts'); from bl_scene import *
s, groups, c = build()
for g,d in groups.items():
    d['P'].inputs['Base Color'].default_value = (*(GOLD if g=='clavicle' else IVORY),1)
    d['M'].inputs[0].default_value = 1.0 if g in ('clavicle','sternum','ribs','cartilage','vertebrae','humerus','scapula') else 1.0
    if g not in ('clavicle',): d['M'].inputs[0].default_value = 0.35
tgt = mathutils.Vector((-80, -110, 1150)); az, el, dist = math.radians(-20), math.radians(8), 1500
c.location = tgt + mathutils.Vector((dist*math.sin(az)*math.cos(el), -dist*math.cos(az)*math.cos(el), dist*math.sin(el)))
c.rotation_euler = (tgt - c.location).to_track_quat('-Z','Y').to_euler()
s.render.filepath = '/home/user/work/b3d/prev_' + s.render.engine + '.png'; bpy.ops.render.render(write_still=True)
