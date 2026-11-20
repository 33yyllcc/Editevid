import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
def make(n=4):
    opt = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path='/home/user/work/models/face_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE, num_faces=n, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
    return vision.FaceLandmarker.create_from_options(opt)
def det_crop(lm, img, x0, y0, s, out=None):
    """run on square crop [x0,y0,s] (can extend outside image; padded) resized to 512"""
    H,W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT)
    c = pad[y0+s:y0+2*s, x0+s:x0+2*s]
    sc = 512/s
    c = cv2.resize(c, (512,512), interpolation=cv2.INTER_CUBIC if sc>1 else cv2.INTER_AREA)
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
    res=[]
    for f in r.face_landmarks:
        res.append(np.array([[l.x*s+x0, l.y*s+y0, l.z*s] for l in f]))
    return res
def skin_score(img, p):
    H,W=img.shape[:2]; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sc=[]
    for i in (50,280,101,330,151,9,4):  # cheeks, forehead, nose
        x,y=int(p[i,0]),int(p[i,1])
        if 0<=x<W and 0<=y<H:
            h,s,v=hsv[max(0,y-2):y+3,max(0,x-2):x+3].reshape(-1,3).mean(0)
            sc.append(1.0 if ((h<=22 or h>=170) and 45<=s<=170 and v>=70) else 0.0)
    return np.mean(sc) if sc else 0
def find_face(lm, img, sizes=(360,540)):
    H,W=img.shape[:2]; cands=[]
    for s in sizes:
        for y0 in range(0, max(1,H-s+1), s//2):
            for x0 in range(0, W-s+1, s//3):
                for p in det_crop(lm, img, x0, y0, s):
                    cands.append((skin_score(img,p), p))
    if not cands: return None, 0
    best=max(cands, key=lambda c:(c[0], np.ptp(c[1][:,0])))
    return best[1], best[0]
