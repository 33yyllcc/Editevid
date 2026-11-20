import sys, glob, cv2, numpy as np
sys.path.insert(0,'/home/user/work/scripts'); from facefind import *
lm=make(); ims=[]
for f in sorted(glob.glob('/home/user/work/frames/s/*.jpg')):
    img=cv2.imread(f); p,sc=find_face(lm,img)
    k=f.split('/')[-1]
    if p is not None:
        x,y=p[:,:2].min(0).astype(int); w,h=np.ptp(p[:,:2],0).astype(int)
        print(k,'score',sc,(x,y,w,h)); cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),6)
    else: print(k,'none')
    ims.append(cv2.resize(img,(480,270)))
ims+= [np.zeros_like(ims[0])]*(16-len(ims))
cv2.imwrite('/home/user/work/frames/check2.jpg', np.vstack([np.hstack(ims[i:i+4]) for i in range(0,16,4)]))
