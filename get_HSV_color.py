import numpy as np
import cv2 
import glob
import csv 

class PointList():
    def __init__(self, npoints):
        self.npoints = npoints
        self.ptlist = np.empty((npoints, 2), dtype=int)
        self.pos = 0 

    def add(self, x, y): 
        self.ptlist[self.pos, :] = [x, y]
        self.pos += 1
                


def onMouse(event, x, y, flag, params):
    global thresh_gbr
    global thresh_list
    thresh_gbr = np.zeros(3)
    img, ptlist = params
    h,w,_ = img.shape[:3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    if event == cv2.EVENT_RBUTTONDOWN:
        thresh_hsv= img[y][x]
        print(thresh_hsv)

if __name__ == '__main__':
    img = cv2.imread("sample.jpg")
    cv2.namedWindow("test")
    npoints = 200 
    ptlist = PointList(npoints)
    cv2.setMouseCallback("test", onMouse, [img, ptlist])
    cv2.imshow("test",img)
    cv2.waitKey(0)                                                                                                              
    
    cv2.destroyAllWindows()

