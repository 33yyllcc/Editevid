import bpy, math, mathutils
def setup(res=900, engine='BLENDER_EEVEE', samples=24):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = engine; s.render.resolution_x = s.render.resolution_y = res
    s.render.film_transparent = True; s.render.image_settings.file_format = 'PNG'; s.render.image_settings.color_mode = 'RGBA'
    s.view_settings.view_transform = 'Filmic' if 'Filmic' in [i.identifier for i in s.view_settings.bl_rna.properties['view_transform'].enum_items] else 'Standard'
    s.view_settings.look = 'None'
    if engine == 'CYCLES':
        s.cycles.samples = samples; s.cycles.use_denoising = False; s.cycles.device = 'CPU'
        s.cycles.max_bounces = 3; s.cycles.diffuse_bounces = 1; s.cycles.glossy_bounces = 1; s.cycles.transmission_bounces = 0; s.cycles.transparent_max_bounces = 6
        s.cycles.use_adaptive_sampling = True; s.cycles.adaptive_threshold = 0.05; s.render.use_persistent_data = True; s.cycles.caustics_reflective = False; s.cycles.caustics_refractive = False
        try: s.cycles.denoiser = 'OPENIMAGEDENOISE'
        except Exception: pass
    if engine == 'BLENDER_EEVEE':
        e = s.eevee; e.taa_render_samples = samples; e.use_gtao = True; e.gtao_distance = 8; e.use_soft_shadows = True
    w = bpy.data.worlds.new('w'); s.world = w; w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.9, 0.92, 1.0, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.35
    def light(name, typ, loc, energy, size=200):
        l = bpy.data.lights.new(name, typ); l.energy = energy
        if typ == 'AREA': l.size = size
        o = bpy.data.objects.new(name, l); s.collection.objects.link(o); o.location = loc
        d = mathutils.Vector((0, 0, 0)) - o.location; o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return o
    light('key', 'AREA', (-300, -400, 350), 2.2e6, 300)
    light('fill', 'AREA', (400, -300, 50), 0.7e6, 400)
    light('rim', 'AREA', (100, 450, 300), 1.4e6, 250)
    return s
def bone_mat(name='bone', col=(0.86, 0.80, 0.68)):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*col, 1); b.inputs['Roughness'].default_value = 0.55
    for k in ('Subsurface Weight', 'Subsurface'):
        if k in b.inputs: b.inputs[k].default_value = 0.08
    return m
def load(path, mat):
    bpy.ops.wm.stl_import(filepath=path); o = bpy.context.selected_objects[0]
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o
def cam(dist=500, lens=70):
    c = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); bpy.context.scene.collection.objects.link(c)
    c.data.lens = lens; c.data.clip_end = 10000; bpy.context.scene.camera = c
    return c
def load_group(paths, mat, center=None):
    objs = [load(p, mat) for p in paths]
    bpy.context.view_layer.update()
    vs = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    lo = mathutils.Vector([min(v[i] for v in vs) for i in range(3)]); hi = mathutils.Vector([max(v[i] for v in vs) for i in range(3)])
    c = (lo + hi) / 2 if center is None else center
    piv = bpy.data.objects.new('piv', None); bpy.context.scene.collection.objects.link(piv)
    for o in objs:
        o.data.transform(mathutils.Matrix.Translation(-c)); o.location = (0, 0, 0); o.parent = piv
    return piv, objs, (hi - lo), c
