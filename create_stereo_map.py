import numpy as np
import cv2 as cv
import glob


# Find Chessboard Corners
chessboardSize = (9,6)
frameSize = (640,480)

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparing object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 20
print(objp)

# Storing object points and image points
objpoints = [] # 3D point in real world space
img_points_L = [] # 2D points in image plane
img_points_R = [] # 2D points in image plane

images_Left = sorted(glob.glob('images/stereoLeft/*.png'))
images_Right = sorted(glob.glob('images/stereoRight/*.png'))

num = 0

for img_Left, img_Right in zip(images_Left, images_Right):

    img_L = cv.imread(img_Left)
    img_R = cv.imread(img_Right)
    gray_L = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
    gray_R = cv.cvtColor(img_R, cv.COLOR_BGR2GRAY)

    # Find chess board corners
    ret_L, cornersL = cv.findChessboardCorners(gray_L, chessboardSize, None)
    ret_R, cornersR = cv.findChessboardCorners(gray_R, chessboardSize, None)

    # Add object points, image points 
    
    if ret_L and ret_R == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(gray_L, cornersL, (11,11), (-1,-1), criteria)
        img_points_L.append(cornersL)

        cornersR = cv.cornerSubPix(gray_R, cornersR, (11,11), (-1,-1), criteria)
        img_points_R.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(img_L, chessboardSize, cornersL, ret_L)
        cv.imshow('img left', img_L)
        cv.drawChessboardCorners(img_R, chessboardSize, cornersR, ret_R)
        cv.imshow('img right', img_R)

        cv.imwrite('images/calibration/Left/imageL' + str(num) + '.png', img_L)
        cv.imwrite('images/calibration/Right/imageR' + str(num) + '.png', img_R)
        print("Images Saved!")
        num += 1

        cv.waitKey(1000)


cv.destroyAllWindows()

# Calibration

ret_L, cameraMatrix_L, dist_L, rvecs_L, tvecs_L = cv.calibrateCamera(objpoints, img_points_L, frameSize, None, None)
heightL, widthL, channelsL = img_L.shape
newCameraMatrix_L, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrix_L, dist_L, (widthL, heightL), 1, (widthL, heightL))

ret_R, cameraMatrix_R, dist_R, rvecs_R, tvecs_R = cv.calibrateCamera(objpoints, img_points_R, frameSize, None, None)
heightR, widthR, channelsR = img_R.shape
newCameraMatrix_R, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrix_R, dist_R, (widthR, heightR), 1, (widthR, heightR))

# Stereo Vision Calibration 
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Fixing intrinsic camera matrixes
# Thus intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrix_L, dist_L, newCameraMatrix_R, dist_R, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, img_points_L, img_points_R, newCameraMatrix_L, dist_L, newCameraMatrix_R, dist_R, gray_L.shape[::-1], criteria_stereo, flags)

# Stereo Rectification 
rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrix_L, dist_L, newCameraMatrix_R, dist_R, gray_L.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrix_L, dist_L, rectL, projMatrixL, gray_L.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrix_R, dist_R, rectR, projMatrixR, gray_R.shape[::-1], cv.CV_16SC2)

print("Saving Parameters")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()