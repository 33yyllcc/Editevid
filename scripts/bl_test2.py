import bpy, sys, math, mathutils
sys.path.insert(0,'/home/user/work/scripts'); from bl_lib import *
s=setup(600, engine='CYCLES', samples=48); m=bone_mat()
piv,objs,size,c=load_group(['/home/user/work/bones/FMA23130.stl'], m)
print('SIZE',size)
L=max(size); cm=cam(); cm.location=(0,-L*2.8,0); cm.rotation_euler=(math.pi/2,0,0)
for o in bpy.data.objects:
    if o.type=='LIGHT': o.location=o.location*(L/350)
piv.rotation_euler=(0,0,math.radians(30))
s.render.filepath='/home/user/work/b3d/hum.png'; bpy.ops.render.render(write_still=True)
