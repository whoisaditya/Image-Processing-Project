import cv2
import numpy as np
from preprocessing import add_gaussian_noise, wiener_filter, gaussian_kernel

def get_Disparity(img_L,img_R):
    stereo = cv2.StereoBM_create()
    img_L_grey=cv2.cvtColor(img_L,cv2.COLOR_BGR2GRAY)
    img_R_grey=cv2.cvtColor(img_R,cv2.COLOR_BGR2GRAY)

    '''
    img_R_grey=cv2.GaussianBlur(img_R_grey, (3,3),cv2.BORDER_DEFAULT)
    img_R_grey = add_gaussian_noise(img_R_grey, sigma = 15)
    img_R_grey = wiener_filter(img_R_grey, kernel = gaussian_kernel(3), K = 10)
    img_R_grey=img_R_grey.astype(np.uint8)

    img_L_grey=cv2.GaussianBlur(img_L_grey, (3,3),cv2.BORDER_DEFAULT)
    img_L_grey = add_gaussian_noise(img_L_grey, sigma = 15)
    img_L_grey = wiener_filter(img_L_grey, kernel = gaussian_kernel(3), K = 10)
    img_L_grey=img_L_grey.astype(np.uint8)
    '''

    min_disparity=0
    num_disparities=16
    block_size=5
    
    stereo.setNumDisparities(5*16)
    stereo.setBlockSize(3*5)
    stereo.setMinDisparity(0) 
    stereo.setPreFilterType(1)
    stereo.setPreFilterSize(7)
    stereo.setTextureThreshold(17)
    stereo.setUniquenessRatio(1)
    stereo.setSpeckleRange(100)
    stereo.setSpeckleWindowSize(100)
    stereo.setDisp12MaxDiff(100)
    


    disp = stereo.compute(img_L_grey,img_R_grey)
    disp = disp.astype(np.float32)
    disp = (disp/(5*16.0) - min_disparity)/num_disparities
    return disp



