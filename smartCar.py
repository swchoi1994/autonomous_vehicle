import numpy as np
import cv2
import time
import cv2 
import imutils 
import matplotlib.pyplot as plt
#from laneDetection.py import canny_edge_detector

car_cascade = 'cascades/haarcascade_car.xml'
car_classifier = cv2.CascadeClassifier(car_cascade)


pedestrain = cv2.HOGDescriptor() 
pedestrain.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

capture = cv2.VideoCapture('files/test3.mp4')

while capture.isOpened():

    response, frame = capture.read()
    if response:
        frame = imutils.resize(frame, width=min(700, frame.shape[1])) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (regions, _) = pedestrain.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.05) 
        cars = car_classifier.detectMultiScale(gray, 1.2, 3)
        

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
            cv2.putText(frame,"Car Detected",(x,y-5),cv2.FONT_HERSHEY_PLAIN,
                    1,(255,255,255),2,cv2.LINE_AA)

        
        for (x1, y1, w1, h1) in regions:
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 3)
            cv2.putText(frame,"Objects Detected",(x,y-5),cv2.FONT_HERSHEY_PLAIN,
                    1,(0,255,0),2,cv2.LINE_AA)


        cv2.imshow("Smart Car AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()