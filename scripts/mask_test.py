import sys, glob, cv2, numpy as np
sys.path.insert(0,'/home/user/work/scripts'); from facefind import *; from mask import render
lm=make(); crops=[]
for f in sorted(glob.glob('/home/user/work/frames/s/*.jpg')):
    img=cv2.imread(f); p,sc=find_face(lm,img)
    if p is None: continue
    out=render(img,p)
    x,y=p[:,:2].min(0).astype(int); w,h=np.ptp(p[:,:2],0).astype(int)
    s=int(max(w,h)*1.6); cx,cy=x+w//2,y+h//2
    pad=cv2.copyMakeBorder(out,s,s,s,s,cv2.BORDER_CONSTANT)
    crops.append(cv2.resize(pad[cy:cy+2*s, cx:cx+2*s],(360,360)))
    if f.endswith('f740.jpg') or f.endswith('f10.jpg'): cv2.imwrite('/home/user/work/frames/mask_'+f.split('/')[-1], out)
crops+= [np.zeros_like(crops[0])]*(16-len(crops))
cv2.imwrite('/home/user/work/frames/mask_grid.jpg', np.vstack([np.hstack(crops[i:i+4]) for i in range(0,16,4)]))
