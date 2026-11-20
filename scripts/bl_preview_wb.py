import bpy, sys, math, mathutils, time
sys.path.insert(0,'/home/user/work/scripts'); from bl_scene import *
s, groups, c = build()
sh = s.display.shading
sh.light = 'STUDIO'; sh.studio_light = 'paint.sl' if 'paint.sl' in [x for x in bpy.context.preferences.studio_lights.keys()] else sh.studio_light
sh.color_type = 'MATERIAL'; sh.show_shadows = True; sh.shadow_intensity = 0.45; sh.show_cavity = True; sh.cavity_type = 'BOTH'
sh.curvature_ridge_factor = 1.0; sh.curvature_valley_factor = 1.0; sh.show_specular_highlight = True
s.display.render_aa = '16'
print('STUDIO', list(bpy.context.preferences.studio_lights.keys())[:30])
for g,d in groups.items():
    m = d['objs'][0].data.materials[0]
    col = GOLD if g=='clavicle' else IVORY
    m.diffuse_color = (*col, 1.0 if g=='clavicle' else 0.3)
    m.roughness = 0.5
tgt = mathutils.Vector((-80, -110, 1150)); az, el, dist = math.radians(-20), math.radians(8), 1500
c.location = tgt + mathutils.Vector((dist*math.sin(az)*math.cos(el), -dist*math.cos(az)*math.cos(el), dist*math.sin(el)))
c.rotation_euler = (tgt - c.location).to_track_quat('-Z','Y').to_euler()
t=time.time(); s.render.filepath = '/home/user/work/b3d/prev_WB.png'; bpy.ops.render.render(write_still=True); print('RT', time.time()-t)
