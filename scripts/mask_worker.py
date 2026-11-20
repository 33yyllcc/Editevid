import sys, subprocess, cv2, numpy as np, time
sys.path.insert(0,'/home/user/work/scripts')
from facefind import make, det_crop, skin_score, find_face
from mask import render
src, f0, nf, out, log = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
W, H, FPS = 1920, 1080, 30
dec = subprocess.Popen(['ffmpeg','-loglevel','error','-ss',f'{f0/FPS:.6f}','-i',src,'-frames:v',str(nf),'-f','rawvideo','-pix_fmt','bgr24','-'], stdout=subprocess.PIPE)
enc = subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
                        '-c:v','libx264','-preset','fast','-crf','14','-pix_fmt','yuv420p',out], stdin=subprocess.PIPE)
lm2 = make(2)
P = None; S = None; lost = 0; lastP = None; prev_gray = None
LOG = open(log, 'w')
def fsize(p): return float(max(np.ptp(p[:,0]), np.ptp(p[:,1])))
def blur_head(fr, p):
    c = p[:, :2].mean(0); s = fsize(p) * 1.3
    x0, y0 = int(max(c[0]-s, 0)), int(max(c[1]-s, 0)); x1, y1 = int(min(c[0]+s, W)), int(min(c[1]+s, H))
    if x1 - x0 > 4 and y1 - y0 > 4:
        r = fr[y0:y1, x0:x1]; sm = cv2.resize(r, (max(1,(x1-x0)//24), max(1,(y1-y0)//24)))
        fr[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(sm, (x1-x0, y1-y0)), (0,0), 6)
    return fr
t0 = time.time()
for i in range(nf):
    buf = dec.stdout.read(W*H*3)
    if len(buf) < W*H*3: break
    fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    gray = cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2GRAY)
    cut = prev_gray is not None and np.abs(gray.astype(int) - prev_gray).mean() > 28
    prev_gray = gray.astype(int)
    newP = None; status = ''
    if P is not None and not cut:
        c = P[:, :2].mean(0); s = int(max(fsize(P) * 2.4, 160))
        cands = det_crop(lm2, fr, int(c[0]-s/2), int(c[1]-s/2), s)
        best = None
        for q in cands:
            d = np.linalg.norm(q[:, :2].mean(0) - c)
            if d < 0.6*fsize(P) and skin_score(fr, q) >= 0.4 and (best is None or d < best[0]): best = (d, q)
        if best: newP = best[1]; status = 'track'
    if newP is None and (lost % 3 == 0 or cut):
        q, sc = find_face(lm2, fr, sizes=(360, 540, 800))
        if q is not None and sc >= 0.5: newP = q; status = 'search'
    if newP is not None:
        if S is None or status == 'search' or cut: S = newP.copy()
        else:
            mot = np.abs(newP[:, :2] - S[:, :2]).mean() / max(fsize(newP), 1)
            a = float(np.clip(mot / 0.02, 0.35, 1.0))
            S = S + a * (newP - S)
        P = newP; lost = 0; lastP = S.copy()
        fr = render(fr, S)
    else:
        lost += 1
        if lastP is not None and lost <= 45 and not cut:
            fr = render(fr, lastP); fr = blur_head(fr, lastP) if lost > 6 else fr
            status = 'hold' if lost <= 6 else 'hold+blur'
        elif lastP is not None:
            fr = blur_head(fr, lastP); status = 'blur'
        else: status = 'noface'
        if lost > 8: P = None
    LOG.write(f'{f0+i},{status}\n')
    enc.stdin.write(fr.tobytes())
    if i % 300 == 0: print(f'{f0+i} {status} {(time.time()-t0)/(i+1):.3f}s/f', flush=True)
enc.stdin.close(); enc.wait(); LOG.close(); print('DONE', flush=True)
