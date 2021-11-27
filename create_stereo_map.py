import numpy as np
import cv2 as cv
import glob

# Termination criteria
crit = (cv.TERM_crit_EPS + cv.TERM_crit_MAX_ITER, 30, 0.001)

# Find Chessboard Corners
chess_b_size = (9,6)
frame_size = (640,480)

# Preparing object points
objp = np.zeros((chess_b_size[0] * chess_b_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chess_b_size[0],0:chess_b_size[1]].T.reshape(-1,2)

objp = objp * 20
print(objp)

# Storing object points and image points

# 3D point in real world space
objpoints = [] 

# 2D points in image plane
img_p_L = [] 

# 2D points in image plane
img_p_R = [] 

img_L = sorted(glob.glob('images/stereoLeft/*.png'))
imgs_R = sorted(glob.glob('images/stereoRight/*.png'))

num = 0

for img_Left, img_Right in zip(img_L, imgs_R):

    img_L = cv.imread(img_Left)
    img_R = cv.imread(img_Right)
    gray_img_L = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
    gray_img_R = cv.cvtColor(img_R, cv.COLOR_BGR2GRAY)

    # Find chess board corners
    ret_L, con_L = cv.findChessboardCorners(gray_img_L, chess_b_size, None)
    ret_R, con_R = cv.findChessboardCorners(gray_img_R, chess_b_size, None)

    # Add object points, image points 
    # If left and right corners are found
    if ret_L and ret_R == True:

        objpoints.append(objp)

        # Refines the corners in left image
        con_L = cv.cornerSubPix(gray_img_L, con_L, (11,11), (-1,-1), crit)
        img_p_L.append(con_L)

        # Refines the corners in right image
        con_R = cv.cornerSubPix(gray_img_R, con_R, (11,11), (-1,-1), crit)
        img_p_R.append(con_R)

        # Draw and display the corners
        cv.drawChessboardCorners(img_L, chess_b_size, con_L, ret_L)
        cv.imshow('img left', img_L)

        cv.drawChessboardCorners(img_R, chess_b_size, con_R, ret_R)
        cv.imshow('img right', img_R)

        cv.imwrite('images/calibration/Left/imageL' + str(num) + '.png', img_L)
        cv.imwrite('images/calibration/Right/imageR' + str(num) + '.png', img_R)
        print("Images Saved!")
        num += 1

        cv.waitKey(1000)

cv.destroyAllWindows()

# Calibration using calibrateCamera
ret_L, cam_mat_L, dist_L, rvecs_L, tvecs_L = cv.calibrateCamera(objpoints, img_p_L, frame_size, None, None)
h_l, w_L, channelsL = img_L.shape
new_cam_mat_L, roi_L = cv.getOptimalNewcam_mat(cam_mat_L, dist_L, (w_L, h_l), 1, (w_L, h_l))

ret_R, cam_mat_R, dist_R, rvecs_R, tvecs_R = cv.calibrateCamera(objpoints, img_p_R, frame_size, None, None)
h_r, w_r, channelsR = img_R.shape
new_cam_mat_R, roi_R = cv.getOptimalNewcam_mat(cam_mat_R, dist_R, (w_r, h_r), 1, (w_r, h_r))

# Stereo Vision Calibration 
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

# Fixing intrinsic camera matrixes
# Thus intrinsic parameters are the same for both cameras
crit_stereo= (cv.TERM_crit_EPS + cv.TERM_crit_MAX_ITER, 30, 0.001)

# Transformation between the two cameras and calculate Essential and Fundamenatl matrix 
retStereo, new_cam_mat_L, dist_L, new_cam_mat_R, dist_R, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, img_p_L, img_p_R, new_cam_mat_L, dist_L, new_cam_mat_R, dist_R, gray_img_L.shape[::-1], crit_stereo, flags)

# Stereo Rectification 
rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(new_cam_mat_L, dist_L, new_cam_mat_R, dist_R, gray_img_L.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(new_cam_mat_L, dist_L, rectL, projMatrixL, gray_img_L.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(new_cam_mat_R, dist_R, rectR, projMatrixR, gray_img_R.shape[::-1], cv.CV_16SC2)

print("Saving Parameters")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()