import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path='/home/user/work/models/selfie_multiclass.tflite'), output_category_mask=True, output_confidence_masks=True))
cap = cv2.VideoCapture('/home/user/work/src/lecture.mp4')
outs=[]
for t,(cx,cy) in [(467.2,(1230,330)),(735,(1200,350)),(790,(1750,300)),(20,(1140,400))]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000); ok, fr = cap.read()
    s=420; x0,y0=cx-s//2,cy-s//2
    crop = cv2.copyMakeBorder(fr,s,s,s,s,cv2.BORDER_CONSTANT)[y0+s:y0+2*s, x0+s:x0+2*s]
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
    cm = r.category_mask.numpy_view()
    print(t, [int((cm==k).sum()) for k in range(6)], [c.numpy_view().shape for c in r.confidence_masks][:1])
    face = r.confidence_masks[3].numpy_view().squeeze()
    vis = crop.copy(); vis[face>0.5] = (vis[face>0.5]*0.4 + np.array([0,0,255])*0.6).astype(np.uint8)
    body = r.confidence_masks[2].numpy_view().squeeze(); vis[body>0.5] = (vis[body>0.5]*0.4 + np.array([255,0,0])*0.6).astype(np.uint8)
    outs.append(cv2.resize(vis,(360,360)))
cv2.imwrite('/home/user/work/out/seg_test.jpg', np.hstack(outs))
