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
WINDOW_SIZE=(10,1080,3)
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
mid_bar=np.zeros([500,660,3])
mid_bar.fill(150)
window = np.zeros([1080,1600,3],dtype=np.uint8)
window.fill(60)
window[290:790,70:730]=mid_bar
window[290:790,790:1450]=mid_bar
cv2.putText(window, "Project Dashboard ", (80,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
window_cpy=window
# Open both cameras
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(1, cv2.CAP_DSHOW)


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 9               #Distance between the cameras [cm]
f = 8              #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]




# Main program loop with face detector and depth estimation using stereo vision
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while(cap_right.isOpened() and cap_left.isOpened()):

        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()
        window=np.copy(window_cpy)
    ################## CALIBRATION #########################################################

        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    ########################################################################################

        # If cannot catch any frame, break
        if not succes_right or not succes_left:                    
            break

        else:

            start = time.time()
            
            # Convert the BGR image to RGB
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            
            # Process the image and find faces
            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)

            # Convert the RGB image to BGR
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


            ################## CALCULATING DEPTH #########################################################

            center_right = 0
            center_left = 0

            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_right.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_left.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)




            # If no ball can be caught in one camera show text "TRACKING LOST"
            if not results_right.detections or not results_left.detections:
                cv2.putText(window, "TRACKING LOST", (800,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(window, "TRACKING LOST", (80,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            else:
                
                # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                # All formulas used to find depth is in video presentaion
                depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

                cv2.putText(window, "Distance: " + str(round(depth,1))+" cm", (800,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
                cv2.putText(window, "Distance: " + str(round(depth,1))+" cm", (80,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
                # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
                # print("Depth: ", str(round(depth,1)))



            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            #print("FPS: ", fps)

            cv2.putText(window, f'FPS: {int(fps)}', (800,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(window, f'FPS: {int(fps)}', (80,240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   
            cv2.putText(window, 'Right Frame', (800,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(window, 'Left Frame', (80,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            # Show the frames
            window[300:300+480,80:80+640]=frame_left
            window[300:300+480,800:800+640]=frame_right
            Hori = np.concatenate((frame_right, frame_left), axis=1)
            cv2.imshow("Images", window) 
           


            # Hit "q" to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()