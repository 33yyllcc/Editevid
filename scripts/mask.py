import cv2, numpy as np
from scipy.spatial import Delaunay
REF = np.load('/home/user/work/models/ref_lm.npy')[:, :2]

def L(a, b=None, t=0.0):
    return (a, b, t)
# outline: top edge image-left -> right, then bottom right -> left
TOP = [L(234,93,.5), L(116,123,.5), L(117,50,.45), L(118,101,.42), L(100,120,.05), L(197,6,.15),
       L(329,349,.05), L(347,330,.42), L(346,280,.45), L(345,352,.5), L(454,323,.5)]
BOT = [L(323,361,.4), L(361), L(288), L(397), L(365), L(379), L(378), L(400), L(377), L(152), L(148), L(176), L(149), L(150), L(136), L(172), L(58), L(132), L(93,132,.4)]
INNER = [L(1), L(4), L(5), L(195), L(50), L(280), L(205), L(425), L(207), L(427), L(0), L(17), L(200), L(175), L(61), L(291), L(214), L(434), L(169), L(394)]
PROTECT = [1,4,5,0,13,14,17,61,291,37,267,84,314,152,175,199,200,18,98,327,2,164,48,278,64,294,129,358,148,377,176,400]

def pts(lm, spec, push=None):
    out = []
    for a, b, t in spec:
        p = lm[a] if b is None else lm[a] * (1 - t) + lm[b] * t
        out.append(p)
    out = np.array(out, np.float64)
    return out

def outline(lm):
    top = pts(lm, TOP); bot = pts(lm, BOT)
    # push bottom outward from nose-tip centre (mask stands off the jaw/chin)
    c = lm[4]
    fw = np.linalg.norm(lm[454] - lm[234])
    d = bot - c; n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    bot = bot + d / n * fw * 0.035
    # chin extra drop along nose->chin direction
    down = (lm[152] - lm[6]); down /= np.linalg.norm(down) + 1e-6
    w = np.exp(-((np.arange(len(BOT)) - BOT.index(L(152))) / 3.0) ** 2)
    bot = bot + down[None] * fw * 0.05 * w[:, None]
    # top edge slightly raised at nose bridge (wire)
    return top, bot

def ctrl(lm):
    top, bot = outline(lm)
    return np.vstack([top, bot, pts(lm, INNER)])

# ---------------- texture in reference space -----------------
SC = 6.0
_rt, _rb = outline(REF)
_RC = ctrl(REF)
OFF = _RC.min(0) - 25
TW, TH = (np.ceil((_RC.max(0) - OFF + 25) * SC)).astype(int)
TRI = Delaunay(_RC).simplices

def _spline(poly, n=12):
    P = np.vstack([poly[-1], poly, poly[0], poly[1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)

def make_texture():
    rng = np.random.default_rng(3)
    T = lambda p: (p - OFF) * SC
    poly = _spline(T(np.vstack([_rt, _rb])))
    m = np.zeros((TH, TW), np.uint8); cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    cx = T(REF[4])[0]; top_y = T(REF[197])[1]; chin_y = T(REF[152])[1]
    hw = (T(REF[454])[0] - T(REF[234])[0]) / 2
    u = (xx - cx) / hw                      # -1..1 across
    v = (yy - top_y) / (chin_y - top_y)     # 0 top .. 1 chin
    def sstep(x, a, b):
        t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
    hgt = -0.55 * u ** 2
    wv = 0.10 + 0.32 * np.clip(v, 0, 1)
    hgt += 0.42 * np.exp(-np.abs(u) / wv) * (1 - sstep(v, 0.2, 0.75))
    hgt += 0.16 * np.exp(-(u / 0.55) ** 2 - ((v - 0.62) / 0.28) ** 2)
    hgt -= 0.6 * sstep(v, 0.8, 1.15) ** 2
    for vp in (0.42, 0.57, 0.72):
        curve = vp + 0.05 * u ** 2 + 0.008 * np.sin(u * 7)
        d = (v - curve) / 0.035
        hgt += 0.018 * (np.tanh(-d * 2.5) * 0.5 + 0.5) * (1 - 0.5 * np.abs(u))
    hgt = cv2.GaussianBlur(hgt.astype(np.float32), (0, 0), 0.8 * SC)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8 * hw
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8 * hw
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Ld = np.array([-0.15, -0.45, 1.0]); Ld /= np.linalg.norm(Ld)
    Hd = Ld + np.array([0, 0, 1.0]); Hd /= np.linalg.norm(Hd)
    diff = np.clip((n * Ld).sum(-1), 0, 1); spec = np.clip((n * Hd).sum(-1), 0, 1) ** 14
    base = 0.035 + 0.085 * diff + 0.05 * spec
    # fabric weave + soft noise
    n1 = rng.normal(0, 1, (TH // 3 + 1, TW // 3 + 1)).astype(np.float32)
    n1 = cv2.resize(cv2.GaussianBlur(n1, (0, 0), 1.2), (TW, TH))[:TH, :TW]
    weave = 0.5 + 0.5 * np.sin(xx * 0.9) * np.sin(yy * 0.9)
    base += 0.008 * n1 + 0.004 * weave
    # edge binding band
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    bw = 2.2 * SC
    band = (dist < bw).astype(np.float32)
    base = base * (1 - 0.25 * band) + 0.025 * np.exp(-((dist - bw) / (0.6 * SC)) ** 2)
    stitch = (np.abs(dist - bw * 0.55) < 0.25 * SC) & ((np.floor((xx + yy) / (1.6 * SC)) % 2) == 0)
    base += 0.02 * stitch
    # nose wire bump along top centre
    base += 0.03 * np.exp(-((dist - 3.5 * SC) / (1.2 * SC)) ** 2) * np.exp(-(u / 0.35) ** 2) * (v < 0.3)
    tex = np.clip(base, 0, 1)
    col = np.stack([tex * 1.04, tex * 1.0, tex * 0.98], -1)  # very slight cool tint (BGR)
    alpha = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.8 * SC)
    return (col * 255).astype(np.float32), alpha

TEX, TALPHA = make_texture()
REFT = (_RC - OFF) * SC

def render(frame, lm, lum_ref=None):
    """frame BGR uint8, lm (478,2+) landmarks in frame px. returns new frame."""
    lm = lm[:, :2]
    C = ctrl(lm)
    H, W = frame.shape[:2]
    x0, y0 = np.floor(C.min(0) - 20).astype(int); x1, y1 = np.ceil(C.max(0) + 20).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    if x1 - x0 < 4 or y1 - y0 < 4: return frame
    rw, rh = x1 - x0, y1 - y0
    col = np.zeros((rh, rw, 3), np.float32); al = np.zeros((rh, rw), np.float32)
    for tri in TRI:
        src = REFT[tri].astype(np.float32); dst = (C[tri] - [x0, y0]).astype(np.float32)
        M = cv2.getAffineTransform(src, dst)
        tm = np.zeros((rh, rw), np.uint8)
        cv2.fillConvexPoly(tm, np.round(dst * 8).astype(np.int32), 255, cv2.LINE_AA, 3)
        wc = cv2.warpAffine(TEX, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wa = cv2.warpAffine(TALPHA, M, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0)
        sel = tm > 0
        col[sel] = wc[sel]; al[sel] = np.maximum(al[sel], wa[sel])
    # protect: lips/nose/chin always covered
    prot = cv2.convexHull((lm[PROTECT] - [x0, y0]).astype(np.float32)).astype(np.int32)
    pm = np.zeros((rh, rw), np.uint8); cv2.fillConvexPoly(pm, prot, 255)
    fw = float(np.ptp(lm[:, 0]))
    dk = int(max(5, fw * 0.07)) | 1
    pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)))
    pm = cv2.GaussianBlur(pm.astype(np.float32) / 255, (0, 0), 1.0)
    need = (pm > 0.01) & (al < pm)
    if need.any():
        fill = col.copy(); empty = col.sum(-1) == 0
        for _ in range(6):
            if not (empty & need).any(): break
            fill = np.where(empty[..., None], cv2.dilate(fill, np.ones((7, 7), np.uint8)), fill); empty = fill.sum(-1) == 0
        fill[empty] = 26
        col = np.where(((col.sum(-1) == 0) & need)[..., None], fill, col)
        al = np.maximum(al, pm)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    # lighting match: scale by local face brightness (forehead) vs reference
    fh = lm[[151, 9, 108, 337]].mean(0).astype(int)
    fy, fx = np.clip(fh[1], 2, H - 3), np.clip(fh[0], 2, W - 3)
    lum = frame[fy - 2:fy + 3, fx - 2:fx + 3].astype(np.float32).mean()
    k = np.clip(lum / (lum_ref or 190.0), 0.6, 1.25)
    c2 = col * k
    bs = max(0.5, rw / 260)
    c2 = cv2.GaussianBlur(c2, (0, 0), bs); al = cv2.GaussianBlur(al, (0, 0), bs * 0.8)
    # contact shadow on skin just above/around mask
    sh = cv2.GaussianBlur(al, (0, 0), max(2.0, rw / 90))
    roi = roi * (1 - 0.28 * np.clip(sh - al, 0, 1)[..., None])
    # grain to match camera noise
    c2 += np.random.normal(0, 2.2, c2.shape).astype(np.float32)
    a = al[..., None]
    out = roi * (1 - a) + c2 * a
    frame = frame.copy(); frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return frame
