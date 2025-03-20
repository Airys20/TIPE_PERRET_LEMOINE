import cv2 as cv
import numpy as np 

def coordonnees_image (path) :
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (1,1), cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(blur, 100, 155, cv.THRESH_BINARY_INV)
    contours, hierarchies = cv.findContours( thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.circle(img, (cx, cy), 7, (0, 0, 255), -1)
        print(f"x: {cx} y: {cy}")
    
    cv.imwrite("etoiles_center.png", img)

coordonnees_image("etoiles_20-02_v2.jpg")