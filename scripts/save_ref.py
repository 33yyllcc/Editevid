import sys, cv2, numpy as np
sys.path.insert(0,'/home/user/work/scripts'); from facefind import *
lm=make(1); img=cv2.imread('/home/user/work/frames/s/f10.jpg')
# crop tightly around known face and detect
ps=det_crop(lm,img,1000,270,300)
p=ps[0]; np.save('/home/user/work/models/ref_lm.npy',p); print(p.shape, p[1], p[152], p[234], p[454])
c=img[270:570,1000:1300].copy()
big=cv2.resize(c,(1200,1200))
for i,(x,y,z) in enumerate(p):
    X,Y=int((x-1000)*4),int((y-270)*4)
    cv2.circle(big,(X,Y),2,(0,255,0),-1)
    cv2.putText(big,str(i),(X+2,Y),0,0.28,(0,0,255),1)
cv2.imwrite('/home/user/work/frames/ref_idx.jpg',big[200:1100,150:1050])
