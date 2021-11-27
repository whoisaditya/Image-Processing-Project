import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
import mediapipe as mp
import time

# Creating the Dashboard
WINDOW_SIZE=(10,1080,3)
facedect = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
mid_bar=np.zeros([500,660,3])
mid_bar.fill(150)
window = np.zeros([1080,1600,3],dtype=np.uint8)
window.fill(60)
window[290:790,70:730]=mid_bar
window[290:790,790:1450]=mid_bar
cv2.putText(window, "Project Dashboard ", (80,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
window_cpy=window
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Frame rate
frame_rate = 120

# Distance between cameras
B = 11               

# focal length of cameras
f = 20               
 
# feild of view of cameras
alpha = 56.6 

with facedect.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while(cap_right.isOpened() and cap_left.isOpened()):
        isLeft, frame_left = cap_left.read()
        isRight, frame_right = cap_right.read()
       
        window=np.copy(window_cpy)
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
        if not isRight or not isLeft:                    
            break

        else:

            start = time.time()
            
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            
            faces_right = face_detection.process(frame_right)
            faces_left = face_detection.process(frame_left)

            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            
            centerL = 0

            centerR = 0
            

            if faces_right.detections:
                for id, detection in enumerate(faces_right.detections):
                    mp_draw.draw_detection(frame_right, detection)
                    bnd_box = detection.location_data.relative_bounding_box
                    hR, wR, cR = frame_right.shape
                    X_min=bnd_box.xmin
                    Y_min=bnd_box.ymin
                    bd_width=bnd_box.width
                    bd_height=bnd_box.height
                    boundBox = int(X_min * wR), int(Y_min * hR), int(bd_width * wR), int(bd_height * hR)
                    x=boundBox[0]
                    w=boundBox[2]
                    y=boundBox[1]
                    h=boundBox[3]
                    center_pt_R = (x + w/ 2, y + h / 2)
                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            if faces_left.detections:
                for id, detection in enumerate(faces_left.detections):
                    mp_draw.draw_detection(frame_left, detection)
                    bnd_box = detection.location_data.relative_bounding_box
                    hL, wL, cL = frame_left.shape
                    X_min=bnd_box.xmin
                    Y_min=bnd_box.ymin
                    bd_width=bnd_box.width
                    bd_height=bnd_box.height
                    boundBox = int(X_min * wL), int(Y_min * hL), int(bd_width * wL), int(bd_height * hL)
                    x=boundBox[0]
                    w=boundBox[2]
                    y=boundBox[1]
                    h=boundBox[3]
                    center_pt_L = (x + w/ 2, y + h / 2)
                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)




            if not faces_right.detections or not faces_left.detections:
                cv2.putText(window, "NO FACE", (800,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(window, "NO FACE", (80,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            else:
                depth = tri.get_depth(center_pt_R, center_pt_L, frame_right, frame_left, B, f, alpha)

                cv2.putText(window, "Distance: " + str(abs(round(depth,2)))+" cm", (800,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
                cv2.putText(window, "Distance: " + str(abs(round(depth,2)))+" cm", (80,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
                
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(window, f'FPS: {int(fps)}', (800,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(window, f'FPS: {int(fps)}', (80,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   
            cv2.putText(window, 'Right Frame', (800,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(window, 'Left Frame', (80,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            window[300:300+480,80:80+640]=frame_left
            window[300:300+480,800:800+640]=frame_right
            cv2.imshow("Images", window) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap_right.release()
cap_left.release()

cv2.destroyAllWindows()