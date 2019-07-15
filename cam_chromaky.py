import cv2
import numpy as np

cap = cv2.VideoCapture(0)
img =  cv2.imread("syc_2.jpg")
rimg = cv2.resize(img,(640,480))
kernel = np.ones((3,3),np.uint8)
while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([70,50,50])
    upper_green = np.array([100,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    ms = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= ms)
    res2 = cv2.bitwise_and(rimg,rimg, mask = ms)
    frame -=res
    frame +=res2
    cv2.imshow('frame',frame)
    cv2.imshow('mask',ms)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
