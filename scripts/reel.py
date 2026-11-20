import sys, subprocess, numpy as np, cv2, os
sys.path.insert(0, '/home/user/work/scripts')
import compose as C
SRC = '/home/user/work/out/m2_full.mp4'; OUT = '/home/user/work/out/reel_video.mp4'
SEGS = [('clavicle', 733.0, 742.0), ('cage', 749.0, 766.0), ('scapula', 768.8, 776.0), ('humerus', 777.6, 786.0), ('forearm', 803.3, 815.8), ('hand', 816.8, 834.0)]
RW, RH, FPS = 1080, 1920, 30
yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
r = np.sqrt(((xx - 540) / 900) ** 2 + ((yy - 640) / 1100) ** 2)
c0 = np.array([46, 34, 24], np.float32); c1 = np.array([20, 13, 9], np.float32)
BGV = np.clip(c0 * (1 - np.clip(r, 0, 1)[..., None]) + c1 * np.clip(r, 0, 1)[..., None] + ((xx % 48 < 2) & (yy % 48 < 2))[..., None] * 10, 0, 255).astype(np.uint8)
HOOK = C.chip('Upper Limb Bones in 1 minute', 'كل عظام الطرف العلوي بدقيقة', 1.35)
END = C.chip('Full lesson on YouTube', 'الشرح الكامل على يوتيوب', 1.3)
enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{RW}x{RH}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
tl = 0.0; total = sum(b - a for _, a, b in SEGS)
for (k, a, b) in SEGS:
    n = int(round((b - a) * FPS))
    dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-ss', f'{a:.4f}', '-i', SRC, '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    for i in range(n):
        buf = dec.stdout.read(1920 * 1080 * 3)
        if len(buf) < 1920 * 1080 * 3: break
        fr = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).copy(); t = a + i / FPS
        comp = C.compose_clip(fr.copy(), t, k)
        panel = comp[20:1010, 0:724]
        out = BGV.copy()
        pn = cv2.resize(panel, (int(724 * 1.15), int(990 * 1.15)), interpolation=cv2.INTER_CUBIC)
        # feather panel borders into bg
        ph, pw = pn.shape[:2]; m = np.ones((ph, pw), np.float32)
        m *= np.clip(np.minimum(np.arange(pw) / 40, (pw - 1 - np.arange(pw)) / 40), 0, 1)[None]; m *= np.clip((ph - 1 - np.arange(ph)) / 60, 0, 1)[:, None]
        x0 = (RW - pw) // 2; y0 = 70
        roi = out[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
        out[y0:y0 + ph, x0:x0 + pw] = (roi * (1 - m[..., None]) + pn * m[..., None]).astype(np.uint8)
        vid = cv2.resize(fr[:, 150:1770], (1080, 720), interpolation=cv2.INTER_AREA)
        out[1200:1920] = vid
        cv2.line(out, (0, 1199), (1080, 1199), (64, 176, 255), 4)
        if tl < 3.2:
            pa = C.ease(tl / 0.3) * (1 - C.ease((tl - 2.9) / 0.3)); C.blend(out, HOOK, (RW - HOOK.shape[1]) / 2, 1040 - HOOK.shape[0] / 2, pa)
        if tl > total - 3.0:
            pa = C.ease((tl - (total - 3.0)) / 0.4); C.blend(out, END, (RW - END.shape[1]) / 2, 1040 - END.shape[0] / 2, pa)
        enc.stdin.write(out.tobytes()); tl += 1 / FPS
    dec.stdout.close(); dec.wait()
enc.stdin.close(); enc.wait()
# audio: same segments from final audio with short fades
parts = ''.join(f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS,afade=t=in:d=0.04,afade=t=out:st={b - a - 0.06}:d=0.06[a{i}];" for i, (_, a, b) in enumerate(SEGS))
subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', '/home/user/work/out/final_audio.m4a', '-i', OUT, '-filter_complex',
                parts + ''.join(f'[a{i}]' for i in range(len(SEGS))) + f'concat=n={len(SEGS)}:v=0:a=1[a]', '-map', '1:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest',
                '/home/user/work/out/instagram_reel.mp4'], check=True)
print('REEL DONE', total)
