import cv2
import numpy as np
from matplotlib import pyplot as plt

def getHSVBounds(bgr, dist):
    bgr = np.uint8([[bgr]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) 
    l = hsv[0][0][0] - dist, 100, 100
    u = hsv[0][0][0] + dist, 255, 255
    return [np.array(l), np.array(u)]

# The b,g,r color values of the rings around champion icons on minimap
BLUE = [230, 162, 81]
RED = [51, 62, 231]


img = cv2.imread('test2.png')
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
allyMask = cv2.inRange(hsvImg, *getHSVBounds(BLUE, 5))
enemyMask = cv2.inRange(hsvImg, *getHSVBounds(RED, 5))

circles = cv2.HoughCircles(allyMask,cv2.HOUGH_GRADIENT,1,10,param1 = 30,param2 =15,minRadius = 9, maxRadius = 30)
if(circles is not None):
    print(circles)
    for n in range(circles.shape[1]):
        x = int(circles[0][n][0])
        y = int(circles[0][n][1])
        radius = int(circles[0][n][2])
        cropped = img[y-radius:y+radius,x-radius:x+radius].copy()
        cv2.imshow("yee", cropped)
        key = cv2.waitKey(0)
