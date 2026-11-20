import sys, os, json, subprocess, math, functools, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/user/work/scripts')
from clips import CLIPS, FPS3D
from terms import TERMS
W, H, FPS = 1920, 1080, 30
FD = '/usr/share/fonts/truetype/noto/'
F_EN = lambda s: ImageFont.truetype(FD + 'NotoSans-Bold.ttf', s)
F_AR = lambda s: ImageFont.truetype(FD + 'NotoSansArabic-Bold.ttf', s, layout_engine=ImageFont.Layout.RAQM)
GOLD = (255, 176, 64); WHITE = (255, 255, 255); NAVY = (14, 20, 34)
def ease(x): x = min(1, max(0, x)); return x * x * (3 - 2 * x)
# ---------- caption chip ----------
@functools.lru_cache(None)
def chip(en, ar, scale=1.0):
    fe, fa = F_EN(int(42 * scale)), F_AR(int(32 * scale))
    d = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
    be = d.textbbox((0, 0), en, font=fe); ba = d.textbbox((0, 0), ar, font=fa, direction='rtl', language='ar')
    pw, ph = int(30 * scale), int(18 * scale); bar = int(7 * scale)
    w = max(be[2] - be[0], ba[2] - ba[0]) + 2 * pw + bar; h = (be[3] - be[1]) + (ba[3] - ba[1]) + 2 * ph + int(16 * scale)
    im = Image.new('RGBA', (w + 20, h + 20), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    dr.rounded_rectangle((10, 14, 10 + w, 14 + h), int(16 * scale), fill=(0, 0, 0, 70))
    dr.rounded_rectangle((10, 10, 10 + w, 10 + h), int(16 * scale), fill=(*NAVY, 205))
    dr.rounded_rectangle((10, 10, 10 + bar + 6, 10 + h), int(16 * scale), fill=(*GOLD, 255)); dr.rectangle((10 + bar, 10, 10 + bar + 8, 10 + h), fill=(*NAVY, 205))
    x0 = 10 + bar + pw; y0 = 10 + ph
    dr.text((x0 - be[0], y0 - be[1]), en, font=fe, fill=WHITE)
    y1 = y0 + (be[3] - be[1]) + int(16 * scale)
    dr.text((10 + w - pw - (ba[2] - ba[0]) - ba[0], y1 - ba[1]), ar, font=fa, fill=(*GOLD, 255), direction='rtl', language='ar')
    a = np.array(im).astype(np.float32); return a
def blend(fr, rgba, x, y, alpha=1.0):
    h, w = rgba.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, fr.shape[1]), min(y + h, fr.shape[0])
    if x1 <= x0 or y1 <= y0: return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]; a = src[..., 3:4] / 255 * alpha
    roi = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (roi * (1 - a) + src[..., 2::-1] * a).astype(np.uint8)  # RGBA -> BGR
# ---------- caption schedule ----------
SCHED = []
for i, (t, en, ar) in enumerate(TERMS):
    nxt = TERMS[i + 1][0] if i + 1 < len(TERMS) else 1e9
    SCHED.append((t, min(t + 3.4, nxt - 0.15), en, ar))
def in_clip(t):
    for k, c in CLIPS.items():
        a, b = c['t']
        if a - 0.6 <= t <= b + 0.6: return k
    return None
def draw_caption(fr, t):
    for (a, b, en, ar) in SCHED:
        if a - 0.05 <= t <= b + 0.35:
            p = ease((t - a) / 0.35) * (1 - ease((t - b) / 0.35))
            im = chip(en, ar); h = im.shape[0]
            blend(fr, im, 54, H - 70 - h + (1 - p) * 24, p)
# ---------- split screen ----------
BG = None
def make_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - 380) / 1100) ** 2 + ((yy - 520) / 800) ** 2)
    c0 = np.array([46, 34, 24], np.float32); c1 = np.array([20, 13, 9], np.float32)  # BGR navy
    bg = c0 * (1 - np.clip(r, 0, 1)[..., None]) + c1 * np.clip(r, 0, 1)[..., None]
    # faint dot grid
    g = ((xx % 48 < 2) & (yy % 48 < 2)).astype(np.float32)[..., None] * 10
    return np.clip(bg + g, 0, 255).astype(np.uint8)
PIP_S, PIP_X, PIP_Y = 0.6, 1920 - 1152 - 40, 216
P3X, P3Y = 8, 135
@functools.lru_cache(maxsize=None)
def round_mask(w, h, r):
    m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return (m.astype(np.float32) / 255)[..., None]
@functools.lru_cache(maxsize=64)
def load3d(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None: return None
    rgb = a[..., :3].astype(np.float32); al = a[..., 3:4].astype(np.float32) / 255
    # light denoise on premultiplied colour
    pm = (rgb * al).astype(np.uint8)
    pm = cv2.bilateralFilter(pm, 5, 18, 3).astype(np.float32)
    al2 = cv2.GaussianBlur(al, (0, 0), 0.6)
    if al2.ndim == 2: al2 = al2[..., None]
    h, w = al2.shape[:2]
    fy = np.clip(np.minimum(np.arange(h) / 40, (h - 1 - np.arange(h)) / 90), 0, 1)
    fx = np.clip(np.minimum(np.arange(w) / 30, (w - 1 - np.arange(w)) / 60), 0, 1)
    fe = (fy[:, None] * fx[None, :])[..., None]
    return pm * fe, np.clip(al2 * fe, 0, 1)
PROJ = {}
ZOOM = {'views': 1.45, 'features': 1.45}
def proj(k):
    if k not in PROJ:
        p = f'/home/user/work/b3d/{k}/proj.json'; PROJ[k] = json.load(open(p)) if os.path.exists(p) else None
    return PROJ[k]
def label_img(en, ar, size=1.0, align='left'):
    return chip(en, ar, 0.78 * size)
def arrow(fr, p0, p1, col=(64, 176, 255), th=4):
    cv2.arrowedLine(fr, tuple(map(int, p0)), tuple(map(int, p1)), col, th, cv2.LINE_AA, tipLength=0.22)
HEAD = {'clavicle': ('Upper Limb Bones', 'عظام الطرف العلوي'), 'cage': ('Thoracic Cage', 'القفص الصدري'), 'scapula': ('Upper Limb Bones', 'عظام الطرف العلوي'),
        'humerus': ('Upper Limb Bones', 'عظام الطرف العلوي'), 'forearm': ('Upper Limb Bones', 'عظام الطرف العلوي'), 'hand': ('Bones of the Hand', 'عظام اليد'),
        'position': ('Anatomical Position', 'الوضع التشريحي'), 'directions': ('Anatomical Directions', 'الاتجاهات التشريحية'),
        'views': ('Humerus', 'عظم العضد'), 'features': ('Humerus', 'عظم العضد')}
def compose_clip(fr, t, k):
    c = CLIPS[k]; a, b = c['t']
    p = ease((t - (a - 0.6)) / 0.6) * (1 - ease((t - b) / 0.6))
    if p <= 0.001: return fr
    global BG
    if BG is None: BG = make_bg()
    s = 1 - (1 - PIP_S) * p; x = PIP_X * p; y = PIP_Y * p
    w, h = int(round(W * s)), int(round(H * s))
    small = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
    out = BG.copy()
    # 3D panel
    pr = proj(k); ff = (min(max(t, a), b) - a) * FPS3D; f0i = int(ff); fr_ = ff - f0i; f = int(round(ff))
    img = load3d(f'/home/user/work/b3d/{k}/{f0i:05d}.png'); img2 = load3d(f'/home/user/work/b3d/{k}/{f0i + 1:05d}.png')
    if img is not None and img2 is not None and fr_ > 0.01:
        img = (img[0] * (1 - fr_) + img2[0] * fr_, img[1] * (1 - fr_) + img2[1] * fr_)
    Z = ZOOM.get(k, 1.0)
    if img is not None and Z != 1.0:
        ph0, pw0 = img[0].shape[:2]
        def zc(a_):
            b_ = cv2.resize(a_, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
            if b_.ndim == 2: b_ = b_[..., None]
            yo, xo = (b_.shape[0] - ph0) // 2, (b_.shape[1] - pw0) // 2
            return b_[yo:yo + ph0, xo:xo + pw0]
        img = (zc(img[0]), np.clip(zc(img[1]), 0, 1))
    if img is not None:
        pm, al = img; ph, pw = pm.shape[:2]
        roi = out[P3Y:P3Y + ph, P3X:P3X + pw].astype(np.float32)
        out[P3Y:P3Y + ph, P3X:P3X + pw] = (roi * (1 - al * p) + pm * p).astype(np.uint8)
    # video card with shadow + rounded corners
    r = int(26 * p) + 1
    xi, yi = int(round(x)), int(round(y))
    if p > 0.01:
        sh = np.zeros((H, W), np.float32); cv2.rectangle(sh, (xi + 8, yi + 14), (xi + w + 8, yi + h + 14), 1, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 18)[..., None] * 0.55 * p
        out = (out.astype(np.float32) * (1 - sh)).astype(np.uint8)
    m = round_mask(w, h, r) if p > 0.01 else np.ones((h, w, 1), np.float32)
    x1, y1 = min(xi + w, W), min(yi + h, H)
    roi = out[yi:y1, xi:x1].astype(np.float32); mm = m[:y1 - yi, :x1 - xi]
    out[yi:y1, xi:x1] = (roi * (1 - mm) + small[:y1 - yi, :x1 - xi] * mm).astype(np.uint8)
    if p < 0.999 and p > 0:
        # during transition fade background in
        out = cv2.addWeighted(out, 1, out, 0, 0)
    # header
    en, ar = HEAD[k]; hd = chip(en, ar, 0.85); blend(out, hd, 30, 34 - (1 - p) * 20, p)
    if pr is not None and img is not None:
        qa = pr[min(f0i, len(pr) - 1)]; qb = pr[min(f0i + 1, len(pr) - 1)]
        P = {kk: [qa['p'][kk][j] * (1 - fr_) + qb['p'][kk][j] * fr_ for j in range(3)] for kk in qa['p']}
        q = dict(az=qa['az'] * (1 - fr_) + qb['az'] * fr_, p=P)
        to_px = lambda key: (P3X + ((P[key][0] - 0.5) * Z + 0.5) * pw, P3Y + ((P[key][1] - 0.5) * Z + 0.5) * ph)
        items = []
        for (ta, tb, key, len_, lar) in c.get('labels', []):
            if ta - 0.05 <= t <= tb + 0.3:
                pa = ease((t - ta) / 0.4) * (1 - ease((t - tb) / 0.3)) * p
                items.append((to_px(key), len_, lar, pa))
        for (ta, tb, key, len_, lar, side) in c.get('feats', []):
            if ta - 0.05 <= t <= tb + 0.3:
                facing = math.cos(math.radians(q['az'])) * (1 if side == 'front' else -1)
                pa = ease((t - ta) / 0.4) * (1 - ease((t - tb) / 0.3)) * p * ease((facing - 0.3) / 0.4)
                if pa > 0.01: items.append((to_px(key), len_, lar, pa))
        items.sort(key=lambda it: it[0][1]); lasty = -1e9
        for (ax, ay), len_, lar, pa in items:
            im = chip(len_, lar, 0.72); lh, lw = im.shape[:2]
            ly = max(ay - lh / 2, lasty + 8, 170); ly = min(ly, H - lh - 30); lasty = ly + lh
            lx = 24 if ax > 330 else 420
            ex = lx + lw - 10 if lx < ax else lx + 10
            ov = out.copy(); cv2.line(ov, (int(ex), int(ly + lh / 2)), (int(ax), int(ay)), (64, 176, 255), 3, cv2.LINE_AA)
            cv2.circle(ov, (int(ax), int(ay)), 9, (64, 176, 255), -1, cv2.LINE_AA); cv2.circle(ov, (int(ax), int(ay)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            out = cv2.addWeighted(ov, pa, out, 1 - pa, 0)
            blend(out, im, lx, ly, pa)
        if c.get('arrows') == 'dirs':
            ev = [(958.7, 'mid'), (960.4, 'prox'), (962.4, 'med'), (964.9, 'lat'), (969.7, 'dist')]
            def on(key): tt = dict((e[1], e[0]) for e in ev)[key]; return ease((t - tt) / 0.5) * p
            ov = out.copy(); A = on('mid')
            if A > 0:
                p0, p1 = to_px('midline_top'), to_px('midline_bot'); n = 22
                for i in range(n):
                    if i % 2 == 0:
                        u0, u1 = i / n, (i + 1) / n
                        cv2.line(ov, (int(p0[0] + (p1[0] - p0[0]) * u0), int(p0[1] + (p1[1] - p0[1]) * u0)), (int(p0[0] + (p1[0] - p0[0]) * u1), int(p0[1] + (p1[1] - p0[1]) * u1)), (255, 255, 255), 3, cv2.LINE_AA)
                out = cv2.addWeighted(ov, A, out, 1 - A, 0); blend(out, chip('Median Line', 'الخط الناصف', 0.65), p0[0] - 60, p0[1] - 90, A)
            hm = to_px('humerus'); mt = to_px('midline_top'); sh_ = to_px('shoulder'); el = to_px('elbow'); ft = to_px('fingertip')
            for key, a0, a1, en_, ar_, lp in [
                ('med', hm, (hm[0] + (mt[0] - hm[0]) * 0.55, hm[1]), 'Medial', 'إنسي', (hm[0] + 30, hm[1] + 18)),
                ('lat', (hm[0] - 20, hm[1] + 50), (hm[0] - 150, hm[1] + 50), 'Lateral', 'وحشي', (max(10, hm[0] - 260), hm[1] + 70)),
                ('prox', el, (sh_[0] + (el[0] - sh_[0]) * 0.25, sh_[1] + (el[1] - sh_[1]) * 0.25), 'Proximal', 'داني', (max(10, sh_[0] - 250), sh_[1] - 70)),
                ('dist', el, (el[0] + (ft[0] - el[0]) * 0.75, el[1] + (ft[1] - el[1]) * 0.75), 'Distal', 'قاصي', (ft[0] + 40, ft[1] - 120))]:
                A = on(key)
                if A > 0:
                    ov = out.copy(); arrow(ov, a0, a1, (64, 176, 255), 6); out = cv2.addWeighted(ov, A, out, 1 - A, 0)
                    blend(out, chip(en_, ar_, 0.62), lp[0], lp[1], A)
        if c.get('arrows') == 'ap':
            for key, tt, en_, ar_ in (('front', 928.0, 'Anterior', 'أمامي'), ('back', 930.9, 'Posterior', 'خلفي')):
                A = ease((t - tt) / 0.5) * p
                if A > 0:
                    cc = to_px('cage'); pt = to_px(key); d = np.array(pt) - np.array(cc); d = d / (np.linalg.norm(d) + 1e-6)
                    a0 = np.array(pt) + d * 10; a1 = np.array(pt) + d * 150
                    ov = out.copy(); arrow(ov, a0, a1, (64, 176, 255), 7); out = cv2.addWeighted(ov, A, out, 1 - A, 0)
                    im = chip(en_, ar_, 0.62); blend(out, im, min(max(a1[0] - im.shape[1] / 2, 10), 700 - im.shape[1]), a1[1] + 20, A)
        if c.get('views'):
            ca = math.cos(math.radians(q['az']))
            for en_, ar_, sgn in (('Anterior View', 'منظر أمامي', 1), ('Posterior View', 'منظر خلفي', -1)):
                A = ease((ca * sgn - 0.6) / 0.3) * p
                if A > 0: blend(out, chip(en_, ar_, 0.7), 40, 950, A)
    return out
def render_frame(fr, t):
    k = in_clip(t)
    if k: fr = compose_clip(fr, t, k)
    else: draw_caption(fr, t)
    return fr
if __name__ == '__main__':
    src, out, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    n = int(round((t1 - t0) * FPS))
    dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-ss', f'{t0:.4f}', '-i', src, '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', os.environ.get('PRESET', 'medium'), '-crf', os.environ.get('CRF', '18'), '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    ORIG = os.environ.get('ORIG')
    dec2 = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-ss', f'{t0:.4f}', '-i', ORIG, '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE) if ORIG else None
    for i in range(n):
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        b2 = dec2.stdout.read(W * H * 3) if dec2 else None
        t = t0 + i / FPS
        if b2 is not None and len(b2) == W * H * 3 and (t < 9.6 or t >= 1077.93): b = b2
        fr = np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
        enc.stdin.write(render_frame(fr, t).tobytes())
        if i % 900 == 0: print('t', round(t, 1), flush=True)
    enc.stdin.close(); enc.wait(); print('DONE')
