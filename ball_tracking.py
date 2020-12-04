# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

low_yellow = np.array([20, 100, 120])
high_yellow = np.array([60, 250, 250])

pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam

vs = cv2.VideoCapture('sample.mp4')
_, frame = vs.read()

# cv2.imwrite('janix', frame)


h, w, c = frame.shape
print(h, w)
time.sleep(2.0)
score = 0
# keep looping
while True:
    _, frame = vs.read()
    if frame is None:
        break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = cv2.resize(frame, (920, 460), interpolation=cv2.INTER_LINEAR)
    frame = frame[90: h - 400, 150: w - 690]
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # # construct a mask for the color "green", then perform
    # # a series of dilations and erosions to remove any small
    # # blobs left in the mask
    mask = cv2.inRange(hsv, low_yellow, high_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('video1', mask)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        print(center)
        if center[0] ==  10:
            score += 1
            print(score)
        # only proceed if the radius meets a minimum size
        if radius > 2:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    # cv2.imshow('mask', mask)
    start_point = (0, 90)
    end_point = (0, 185)
    frame = cv2.line(frame, start_point, end_point, (255, 0, 0), 9)
    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
