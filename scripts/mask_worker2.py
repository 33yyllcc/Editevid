import sys, subprocess, cv2, numpy as np, time
sys.path.insert(0, '/home/user/work/scripts')
from facefind import make, det_crop, skin_score, find_face
from mask import render, TOP, pts, outline
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
src, f0, nf, out, log = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
W, H, FPS = 1920, 1080, 30
SEG = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path='/home/user/work/models/selfie_multiclass.tflite'), output_confidence_masks=True, output_category_mask=False))
dec = subprocess.Popen(['ffmpeg','-loglevel','error','-ss',f'{f0/FPS:.6f}','-i',src,'-frames:v',str(nf),'-f','rawvideo','-pix_fmt','bgr24','-'], stdout=subprocess.PIPE)
enc = subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
                        '-c:v','libx264','-preset','fast','-crf','14','-pix_fmt','yuv420p',out], stdin=subprocess.PIPE)
lm2 = make(2)
rng = np.random.default_rng(1)
def fsize(p): return float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
def segment_face(fr, hc, fs):
    s = int(max(3.2 * fs, 320)); x0, y0 = int(hc[0] - s / 2), int(hc[1] - s / 2)
    pad = cv2.copyMakeBorder(fr, s, s, s, s, cv2.BORDER_CONSTANT)
    crop = pad[y0 + s:y0 + 2 * s, x0 + s:x0 + 2 * s]
    r = SEG.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    pf = r.confidence_masks[3].numpy_view().squeeze()
    if pf.shape[0] != s: pf = cv2.resize(pf, (s, s))
    m = (pf > 0.45).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return None
    # choose component closest to crop centre weighted by area
    best = None
    V = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(fs * 0.25) | 1,) * 2)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 0.08 * fs * fs: continue
        comp = (lab == i).astype(np.uint8)
        ring = cv2.dilate(comp, k) - comp
        dark = float((V[ring > 0] < 70).mean()) if ring.any() else 0
        if dark < 0.25: continue
        d = np.linalg.norm(cen[i] - [s / 2, s / 2])
        sc = a * dark / (1 + (d / fs) ** 2)
        if best is None or sc > best[0]: best = (sc, i)
    if best is None: return None
    full = np.zeros((H, W), np.uint8)
    comp = (lab == best[1]).astype(np.uint8)
    xs0, ys0 = max(x0, 0), max(y0, 0); xs1, ys1 = min(x0 + s, W), min(y0 + s, H)
    if xs1 <= xs0 or ys1 <= ys0: return None
    full[ys0:ys1, xs0:xs1] = comp[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    return full
def fabric_fill(fr, region, fs, light_dir=0.0):
    """region uint8 mask (0/1) -> paint shaded black fabric with soft edge"""
    ys, xs = np.nonzero(region)
    if len(xs) == 0: return fr
    x0, x1 = max(xs.min() - 12, 0), min(xs.max() + 13, W); y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 13, H)
    R = region[y0:y1, x0:x1].astype(np.float32)
    a = cv2.GaussianBlur(R, (0, 0), max(1.0, fs / 160))
    dist = cv2.distanceTransform((R > 0).astype(np.uint8), cv2.DIST_L2, 5)
    hh, ww = R.shape; yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    v = (yy - ys.min() + y0) / max(1, ys.max() - ys.min())
    shade = 30 + 10 * np.clip(dist / (0.12 * fs), 0, 1) - 12 * np.clip(v - 0.6, 0, 1) + light_dir * (xx / ww - 0.5) * 8
    shade = shade + rng.normal(0, 2.0, shade.shape)
    col = np.stack([shade * 1.03, shade, shade * 0.98], -1)
    roi = fr[y0:y1, x0:x1].astype(np.float32)
    sh = cv2.GaussianBlur(R, (0, 0), max(2.0, fs / 40)); roi *= (1 - 0.22 * np.clip(sh - a, 0, 1))[..., None]
    fr[y0:y1, x0:x1] = np.clip(roi * (1 - a[..., None]) + col * a[..., None], 0, 255).astype(np.uint8)
    return fr
def search_validated(fr):
    cands = []
    for sz in (360, 540, 800):
        for y0 in range(0, max(1, H - sz + 1), sz // 2):
            for x0 in range(0, W - sz + 1, sz // 3):
                for q in det_crop(lm2, fr, x0, y0, sz):
                    if skin_score(fr, q) >= 0.5: cands.append(q)
        if cands: break
    best = None
    for q in cands:
        c = q[:, :2].mean(0); f = fsize(q)
        R = segment_face(fr, c, f)
        if R is None: continue
        ys, xs = np.nonzero(R)
        if np.linalg.norm([xs.mean() - c[0], ys.mean() - c[1]]) > 0.6 * f: continue
        if best is None or f > best[0]: best = (f, q)
    return None if best is None else best[1]
def below_top_region(lm, fs):
    top, bot = outline(lm[:, :2])
    down = lm[152, :2] - lm[6, :2]; down = down / (np.linalg.norm(down) + 1e-6)
    poly = np.vstack([top, top[::-1] + down * fs * 3])
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m
def blur_head(fr, hc, fs):
    s = fs * 1.3
    x0, y0 = int(max(hc[0] - s, 0)), int(max(hc[1] - s, 0)); x1, y1 = int(min(hc[0] + s, W)), int(min(hc[1] + s, H))
    if x1 - x0 > 4 and y1 - y0 > 4:
        r = fr[y0:y1, x0:x1]; sm = cv2.resize(r, (max(1, (x1 - x0) // 24), max(1, (y1 - y0) // 24)))
        fr[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(sm, (x1 - x0, y1 - y0)), (0, 0), 6)
    return fr
P = None; S = None; hc = None; fs = 140.0; lost = 0; prev_gray = None; rho = 0.47; lastS = None
LOG = open(log, 'w'); t0 = time.time()
for i in range(nf):
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    gray = cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2GRAY).astype(int)
    cut = prev_gray is not None and np.abs(gray - prev_gray).mean() > 28
    prev_gray = gray
    newP = None; mode = ''
    if P is not None and not cut:
        c = P[:, :2].mean(0); s = int(max(fsize(P) * 2.4, 160))
        best = None
        for q in det_crop(lm2, fr, int(c[0] - s / 2), int(c[1] - s / 2), s):
            d = np.linalg.norm(q[:, :2].mean(0) - c)
            if d < 0.6 * fsize(P) and skin_score(fr, q) >= 0.4 and (best is None or d < best[0]): best = (d, q)
        if best: newP = best[1]; mode = 'lm'
    if newP is None and (lost % 3 == 0 or cut or hc is None):
        q = search_validated(fr)
        if q is not None:
            qc = q[:, :2].mean(0)
            near = hc is None or cut or lost > 90 or np.linalg.norm(qc - hc) < 2.0 * fs
            if near: newP = q; mode = 'lm-s'
    if newP is not None:
        if S is None or mode == 'lm-s' or cut: S = newP.copy()
        else:
            mot = np.abs(newP[:, :2] - S[:, :2]).mean() / max(fsize(newP), 1)
            S = S + float(np.clip(mot / 0.02, 0.35, 1.0)) * (newP - S)
        P = newP; lost = 0; hc = S[:, :2].mean(0); fs = 0.8 * fs + 0.2 * fsize(S); lastS = S.copy()
    else:
        lost += 1
        if lost > 8: P = None
    R = segment_face(fr, hc, fs) if hc is not None else None
    if newP is not None:
        fr = render(fr, S)
        if R is not None:
            B = below_top_region(S, fs)
            # what is already covered: re-detect dark mask pixels is unreliable; use render polygon region instead
            top, bot = outline(S[:, :2]); cov = np.zeros((H, W), np.uint8)
            from mask import _spline
            cv2.fillPoly(cov, [_spline(np.vstack([top, bot]), 6).astype(np.int32)], 1)
            unc = (R & B & (1 - cov)).astype(np.uint8)
            if unc.sum() > 0.004 * fs * fs:
                unc = cv2.dilate(unc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(fs * 0.06) | 1,) * 2))
                fr = fabric_fill(fr, unc & B, fs); mode += '+fill'
            # learn where mask top sits relative to the face-skin box (for profile fallback)
            ys = np.nonzero(R)[0]
            if len(ys) > 50:
                ytop, ybot = ys.min(), ys.max(); mt = outline(S[:, :2])[0][:, 1].mean()
                rr = (mt - ytop) / max(1, ybot - ytop)
                if 0.25 < rr < 0.7: rho = 0.95 * rho + 0.05 * rr
    elif R is not None and R.sum() > 0.15 * fs * fs:
        ys, xs = np.nonzero(R); ytop, ybot = ys.min(), ys.max()
        yc = int(ytop + rho * (ybot - ytop))
        low = R.copy(); low[:yc] = 0
        pts_ = cv2.findNonZero(low)
        if pts_ is not None and len(pts_) > 20:
            hull = cv2.convexHull(pts_); reg = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(reg, hull, 1)
            reg = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(fs * 0.07) | 1,) * 2))
            reg[:max(yc - int(fs * 0.02), 0)] = 0
            fr = fabric_fill(fr, reg, fs)
            nc = np.array([xs.mean(), ys.mean()]); dv = nc - hc; dn = np.linalg.norm(dv)
            hc = hc + (dv if dn < 0.3 * fs else dv / dn * 0.3 * fs); mode = 'seg'
        else:
            fr = blur_head(fr, hc, fs); mode = 'blur'
    elif hc is not None and lost <= 150:
        fr = blur_head(fr, hc, fs); mode = 'blur'
    else:
        mode = 'none'
    LOG.write(f'{f0 + i},{mode},{None if hc is None else hc.astype(int).tolist()},{fs:.0f},{0 if R is None else int(R.sum())}\n')
    enc.stdin.write(fr.tobytes())
    if i % 300 == 0: print(f'{f0 + i} {mode} {(time.time() - t0) / (i + 1):.3f}s/f', flush=True)
enc.stdin.close(); enc.wait(); LOG.close(); print('DONE', flush=True)
