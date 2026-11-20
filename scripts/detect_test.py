import sys, glob, cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
opt = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path='/home/user/work/models/face_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
lm = vision.FaceLandmarker.create_from_options(opt)
for f in sorted(glob.glob('/home/user/work/frames/s/*.jpg')):
    img = cv2.imread(f)
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if r.face_landmarks:
        p = np.array([[l.x*img.shape[1], l.y*img.shape[0]] for l in r.face_landmarks[0]])
        print(f.split('/')[-1], 'FACE', p.min(0).astype(int), (p.max(0)-p.min(0)).astype(int))
    else: print(f.split('/')[-1], 'none')
