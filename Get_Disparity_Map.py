import cv2
import numpy as np
def get_Disparity(img_L,img_R,L_Map_X,L_Map_Y,R_Map_X,R_Map_Y):
    stereo = cv2.StereoBM_create()
    img_L_grey=cv2.cvtColor(img_L,cv2.COLOR_BGR2GRAY)
    img_R_grey=cv2.cvtColor(img_R,cv2.COLOR_BGR2GRAY)
    L_adg_img=cv2.remap(img_L_grey,L_Map_X,L_Map_Y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
    R_adg_img=cv2.remap(img_R_grey,R_Map_X,R_Map_Y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
    min_disparity=0;
    num_disparities=16;
    block_size=5;
    
    disp=stereo.compute(L_adg_img,R_adg_img)
    disp=disp.astype(np.float32)


