import numpy as np, sys, wave
SR = 44100; dur = float(sys.argv[1]); out = sys.argv[2]
rng = np.random.default_rng(7)
N = int(SR * dur); L = np.zeros(N); R = np.zeros(N)
def hz(m): return 440 * 2 ** ((m - 69) / 12)
# progression (midi), 8 s per chord, 60 BPM feel
prog = [[48, 55, 64, 71], [45, 52, 60, 67], [41, 48, 57, 64], [43, 50, 59, 66], [45, 52, 60, 64], [41, 48, 55, 64], [48, 55, 62, 67], [43, 50, 55, 62]]
CH = 8.0
t_all = np.arange(N) / SR
def env(n, a, r):
    e = np.ones(n); a = int(a * SR); r = int(r * SR)
    e[:a] = np.linspace(0, 1, a) ** 2; e[-r:] *= np.linspace(1, 0, r) ** 2; return e
k = 0; t = 0.0
while t < dur:
    ch = prog[k % len(prog)]; n0 = int(t * SR); n = min(int((CH + 3) * SR), N - n0)
    if n <= 0: break
    tt = np.arange(n) / SR; e = env(n, 2.5, 3.0)
    for j, m in enumerate(ch):
        f = hz(m); det = 1 + rng.uniform(-0.002, 0.002)
        s = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * det * tt) + 0.08 * np.sin(2 * np.pi * 3 * f * tt))
        s *= (0.5 + 0.5 * np.sin(2 * np.pi * (0.07 + 0.03 * j) * tt + j)) * 0.12
        pan = 0.3 + 0.4 * (j / 3)
        L[n0:n0 + n] += s * e * (1 - pan); R[n0:n0 + n] += s * e * pan
    # soft "piano/bell" notes from pentatonic over the chord
    for b in range(3):
        if rng.random() < 0.75:
            bt = t + rng.uniform(0.5, CH - 1); m = rng.choice([72, 74, 76, 79, 81]) - (12 if rng.random() < 0.3 else 0)
            b0 = int(bt * SR); bn = min(int(4 * SR), N - b0)
            if bn > 0:
                tb = np.arange(bn) / SR; f = hz(m)
                s = (np.sin(2 * np.pi * f * tb) + 0.3 * np.sin(2 * np.pi * 2.01 * f * tb) * np.exp(-tb * 3)) * np.exp(-tb * 1.1) * 0.10
                s *= np.minimum(1, tb / 0.01)
                p = rng.uniform(0.3, 0.7); L[b0:b0 + bn] += s * (1 - p); R[b0:b0 + bn] += s * p
    t += CH; k += 1
# reverb: simple feedback delay network
def reverb(x, mix=0.35):
    y = x.copy()
    for d, g in ((1557, .5), (1617, .48), (1491, .46), (1422, .44), (1277, .42), (1356, .4)):
        buf = np.zeros(len(x) + d)
        for i0 in range(0, len(x), d):
            seg = x[i0:i0 + d] + g * buf[i0:i0 + min(d, len(x) - i0)]
            buf[i0 + d:i0 + d + len(seg)] = seg
        y += mix * buf[d:d + len(x)] / 3
    return y
L, R = reverb(L), reverb(R)
# lowpass gently (one-pole) for warmth
def lp(x, fc=3500):
    a = np.exp(-2 * np.pi * fc / SR); y = np.zeros_like(x); acc = 0.0
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x)
L, R = lp(L), lp(R)
st = np.stack([L, R], 1); st /= np.abs(st).max() + 1e-9; st *= 0.8
fade = int(4 * SR); st[:fade] *= np.linspace(0, 1, fade)[:, None]; st[-fade:] *= np.linspace(1, 0, fade)[:, None]
w = wave.open(out, 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((st * 32767).astype(np.int16).tobytes()); w.close(); print('ok', out)
