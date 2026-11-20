import bpy, sys, math, json, mathutils, os
sys.path.insert(0, '/home/user/work/scripts')
from bl_lib import setup
B = '/home/user/work/bones/'
G = {
 'clavicle': ['FMA13322'], 'scapula': ['FMA13395'], 'humerus': ['FMA23130'], 'radius': ['FMA23464'], 'ulna': ['FMA23467'],
 'carpals': ['FMA23725','FMA24435','FMA24437','FMA24439','FMA24441','FMA24443','FMA24446','FMA24448'],
 'metacarpals': ['FMA24464','FMA24466','FMA24468','FMA24470','FMA24472'],
 'phalanges': ['FMA%d' % i for i in range(24450, 24464)],
 'sternum': ['FMA7486','FMA7487','FMA7488'],
 'ribs': ['FMA7857','FMA7882','FMA7909','FMA7957','FMA8066','FMA8175','FMA8229','FMA8283','FMA8364','FMA8445','FMA8531','FMA8533'],
 'cartilage': ['FMA7875','FMA7886','FMA7913','FMA7976','FMA8070','FMA8194','FMA8248'],
 'vertebrae': ['FMA9165','FMA9187','FMA9209','FMA9248','FMA9922','FMA9945','FMA9968','FMA9991','FMA10014','FMA10037','FMA10059','FMA10081'],
}
MIRROR = {'ribs', 'cartilage'}
IVORY = (0.83, 0.77, 0.64); GOLD = (1.0, 0.55, 0.12)
def make_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True; m.blend_method = 'BLEND'
    nt = m.node_tree; P = nt.nodes['Principled BSDF']; out = nt.nodes['Material Output']
    P.inputs['Roughness'].default_value = 0.5
    if 'Subsurface Weight' in P.inputs: P.inputs['Subsurface Weight'].default_value = 0.05
    T = nt.nodes.new('ShaderNodeBsdfTransparent'); M = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(T.outputs[0], M.inputs[1]); nt.links.new(P.outputs[0], M.inputs[2]); nt.links.new(M.outputs[0], out.inputs[0])
    return m, P, M
def build():
    s = setup(int(os.environ.get('RES_W', 960)), engine=os.environ.get('ENGINE', 'CYCLES'), samples=int(os.environ.get('SPP', 20)))
    s.render.resolution_y = int(os.environ.get('RES_H', 1080))
    groups = {}
    for g, ids in G.items():
        m, P, M = make_mat(g); objs = []
        for i in ids:
            bpy.ops.wm.stl_import(filepath=B + i + '.stl'); o = bpy.context.selected_objects[0]
            o.data.materials.append(m); bpy.ops.object.shade_smooth(); objs.append(o)
            if g in MIRROR:
                o2 = o.copy(); o2.data = o.data.copy(); s.collection.objects.link(o2)
                o2.data.transform(mathutils.Matrix.Scale(-1, 4, (1, 0, 0))); o2.data.flip_normals(); objs.append(o2)
        groups[g] = dict(objs=objs, P=P, M=M)
    c = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); s.collection.objects.link(c); s.camera = c
    c.data.lens = 50; c.data.clip_start = 5; c.data.clip_end = 20000
    # lights around the skeleton
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']: bpy.data.objects.remove(o)
    def light(loc, e, size):
        l = bpy.data.lights.new('l', 'AREA'); l.energy = e; l.size = size
        o = bpy.data.objects.new('l', l); s.collection.objects.link(o); o.location = loc
        o.rotation_euler = (mathutils.Vector((-100, -80, 1100)) - o.location).to_track_quat('-Z', 'Y').to_euler()
    light((-700, -1300, 1900), 9e6, 900); light((900, -900, 1200), 3.5e6, 1200); light((200, 1400, 1600), 6e6, 900); light((-900, 600, 900), 2e6, 900)
    return s, groups, c
def centroid(g, groups):
    vs = [o.matrix_world @ v.co for o in groups[g]['objs'] for v in o.data.vertices]
    return sum(vs, mathutils.Vector()) / len(vs), vs
