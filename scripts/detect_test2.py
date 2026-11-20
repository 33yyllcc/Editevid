import glob, cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
opt = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path='/home/user/work/models/face_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
lm = vision.FaceLandmarker.create_from_options(opt)
def det(img, x0, y0, s):
    c = img[y0:y0+s, x0:x0+s]
    c = cv2.copyMakeBorder(c, 0, s-c.shape[0], 0, s-c.shape[1], cv2.BORDER_CONSTANT)
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
    if not r.face_landmarks: return None
    return np.array([[l.x*s+x0, l.y*s+y0] for l in r.face_landmarks[0]])
for f in sorted(glob.glob('/home/user/work/frames/s/*.jpg')):
    img = cv2.imread(f); H,W = img.shape[:2]; found=None
    for s in (360, 540, 800, 1080):
        for y0 in range(0, max(1,H-s+1), s//2):
            for x0 in range(0, W-s+1, s//3):
                p = det(img, x0, y0, s)
                if p is not None: found=(s,p); break
            if found: break
        if found: break
    if found:
        s,p=found; print(f.split('/')[-1], 'FACE tile', s, p.min(0).astype(int), (p.max(0)-p.min(0)).astype(int))
    else: print(f.split('/')[-1], 'none')
