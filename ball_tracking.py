import argparse
import time
from collections import deque

import cv2
import imutils
import numpy as np


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

low_yellow = np.array([20, 130, 160])
high_yellow = np.array([60, 200, 250])
low_red = np.array([150, 50, 0])
high_red = np.array([255, 255, 255])
low_blue = np.array([94, 80, 2])
high_blue = np.array([126, 255, 255])

pts = deque(maxlen=args["buffer"])

vs = cv2.VideoCapture('sample.mp4')
_, frame = vs.read()

h, w, c = frame.shape
time.sleep(2.0)

score = 0
frame_no = 1
backSub = cv2.createBackgroundSubtractorKNN()
# backSub = cv2.createBackgroundSubtractorMOG2()

prev_frame = None

while True:

    _, frame = vs.read()
    if frame is None:
        break

    frame = frame[150: h - 155, 210: w - 350]
    frame = cv2.resize(frame, (460, 260), interpolation=cv2.INTER_LINEAR)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, low_yellow, high_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None
    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and
        #
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # print(center)
        # if center[0] ==  10:
        #     score += 1
        # print(score)
        # only proceed if the radius meets a minimum size
        if radius > 2:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 1)
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
    start_point = (0, 90)
    end_point = (0, 185)
    frame_no = frame_no + 1
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    # if frame_no == 5000:
    #     break

cv2.destroyAllWindows()

# os.remove("data.csv")
# df = pd.DataFrame(list(zip(*[rod_vals])))
# df.to_csv('data.csv', index=False)

# start_plotting()
#
