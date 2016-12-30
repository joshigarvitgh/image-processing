from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler
import argparse
import imutils
import numpy as np
import cv2
import argparse
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty(): raise Exception("your face_cascade is empty. are you sure, the path is correct ?")

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
if eye_cascade.empty(): raise Exception("your eye_cascade is empty. are you sure, the path is correct ?")

video = cv2.VideoCapture(0)
while(video.isOpened()):
	ret, frame = video.read()
	if frame is not None:
		resized = imutils.resize(frame,width=600)
		ratio=frame.shape[0] / float(resized.shape[0])
		blurred = cv2.GaussianBlur(resized, (5, 5), 0)
		gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
		lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
		thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
# find contours in the thresholded image and initialize the
# shape detector
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		sd = ShapeDetector()
		cl = ColorLabeler()

# loop over the contours
		for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
				M = cv2.moments(c)
				#cX = int((M["m10"] / M["m00"]) * ratio)
				#cY = int((M["m01"] / M["m00"]) * ratio)
				shape = sd.detect(c)
				color = cl.label(lab, c)
				print(shape)
				print(color)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
				c = c.astype("float")
				c *= ratio
				c = c.astype("int")
				cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
				#cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.imshow('Video',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

video.release()
cv2.destroyAllWindows()
