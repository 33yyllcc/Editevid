import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath='/home/user/work/bones/FMA13322.stl')
o=bpy.context.selected_objects[0]
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS'); o.location=(0,0,0)
cam=bpy.data.objects.new('cam',bpy.data.cameras.new('cam')); bpy.context.scene.collection.objects.link(cam)
cam.location=(0,-400,0); cam.rotation_euler=(1.5708,0,0); bpy.context.scene.camera=cam
s=bpy.context.scene; s.render.resolution_x=s.render.resolution_y=400
s.render.engine=sys.argv[-1]; s.render.filepath='/home/user/work/b3d/test_'+sys.argv[-1]+'.png'
if s.render.engine=='CYCLES': s.cycles.samples=8
bpy.ops.render.render(write_still=True)
