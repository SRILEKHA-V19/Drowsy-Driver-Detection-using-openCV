# Haar Cascade for Object Detection

import cv2
import numpy as np
import pyautogui

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

ct = 0
ct1 = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV    
    lower_blue = np.array([0,48,80])
    upper_blue = np.array([20,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    #res = cv2.GaussianBlur(res,(10,10),0)
    blur = cv2.GaussianBlur(mask,(15,15),0)

    #Apply threshold
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ##cv2.imshow('frame',mask)
    #cv2.imshow('ret',ret)
    ##cv2.imshow('thresh',thresh)


    
    #epsilon = 0.01*cv2.arcLength(cnt,True)
    #approx = cv2.approxPolyDP(cnt,epsilon,True)
    
        
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        # (img, startingPt, endingPt, Blueline, lineWidth)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        count=0;

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            
            _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            cnt = contours[0]
            if(len(contours))>=0:
                c=max(contours, key=cv2.contourArea)
                (x,y),radius=cv2.minEnclosingCircle(c)
                M=cv2.moments(c)
            else:
                print("Sorry no contour found")
            cnt=c
            if cv2.contourArea(cnt)<=1000:
                continue
            hull = cv2.convexHull(cnt,returnPoints = False)
            defects = cv2.convexityDefects(cnt,hull)
            
            try:
                defects.shape
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(img,start,end,[0,255,0],2)
                    cv2.circle(img,far,5,[0,0,255],-1)
                    count=count+1
                #print(str(cv2.contourArea(cnt,True)))
                if cv2.arcLength(cnt,True)>2000:
                    while ct==0:
                        print("ON")
                        pyautogui.press('space')
                        ct=1
                        ct1=0
                if cv2.arcLength(cnt,True)>500 and cv2.arcLength(cnt,True)<=1500:
                    while ct1==0:
                        print("OFF")
                        pyautogui.press('space')
                        ct1=1
                        ct=0
                        
                #if arc
            except AttributeError:
                print("shape not found")   

    cv2.imshow('mask', mask)
    cv2.imshow('res', res)        
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
