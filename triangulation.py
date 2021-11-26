import sys
import cv2
import numpy as np
import time


def get_depth(right_pt, left_pt, frame_right, frame_left, B,f, alpha):

    # Converting Focal length from mm to pixels
    heightL, widthL, depth_left = frame_left.shape
    heightR, widthR, depth_right = frame_right.shape
    

    if widthR == widthL:
        f_pixel = (widthR * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right cameras not same')

    xR = right_pt[0]
    xL = left_pt[0]

    # Calculate Disparity
    disparity = xL-xR     

    # Depth calc
    zDepth = (B*f_pixel)/disparity           

    return zDepth
