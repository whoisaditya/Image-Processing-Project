import numpy as np
import cv2

cap = cv2.VideoCapture(0) # Video Input from Laptop Webcam
cap1 = cv2.VideoCapture(1) # Video Input from External Webcam

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()

    w_lap = int(cap.get(3))
    h_lap = int(cap.get(4))

    w_cam = int(cap1.get(3))
    h_cam = int(cap1.get(4))

    image = np.zeros(frame.shape, np.uint8)
    s_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    s_frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=.5)

    image[:h_lap//2, :w_lap//2] = s_frame
    image[h_cam//2:, :w_cam//2] = s_frame1
    image[:h_cam//2, w_cam//2:] = s_frame1
    image[h_lap//2:, w_lap//2:] = s_frame

    cv2.imshow('Input', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()