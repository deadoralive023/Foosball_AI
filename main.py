import cv2
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

cap = cv2.VideoCapture('sample.mp4')
FRAME_WINDOW = "Frame"
P1_WINDOW = "Player 1"
P2_WINDOW = "Player 2"
BALL_WINDOW = "Ball"
cv2.namedWindow(FRAME_WINDOW)
cv2.namedWindow(P1_WINDOW)
cv2.namedWindow(P2_WINDOW)
cv2.namedWindow(BALL_WINDOW)

cv2.moveWindow(FRAME_WINDOW, 00, 0);
cv2.moveWindow(BALL_WINDOW, 950,0);
cv2.moveWindow(P1_WINDOW, 00, 700);
cv2.moveWindow(P2_WINDOW, 950, 700);

_, frame = cap.read()

h,w,c = frame.shape

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (920, 460), interpolation=cv2.INTER_LINEAR)
    frame = frame[78 : h-390, 120 : w - 670]

    # frame_arr = np.array(frame)
    blue = [0, 220, 0]

    # Get X and Y coordinates of all blue pixels
    X, Y = np.where(np.all(frame == blue, axis=2))

    print(X, Y)

    # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # # Yellow Color
    # low_yellow = np.array([20, 110, 140])
    # high_yellow = np.array([60, 220, 220])
    # yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    # yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    #
    # #Red Color
    # low_red = np.array([150, 80, 0])
    # high_red = np.array([255, 255, 255])
    # red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    # red = cv2.bitwise_and(frame, frame, mask=red_mask)
    #
    # # Blue color
    # low_blue = np.array([94, 80, 2])
    # high_blue = np.array([126, 255, 255])
    # blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    # blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    #
    # # Green color
    # low_green = np.array([25, 52, 72])
    # high_green = np.array([102, 255, 255])
    # green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    # green = cv2.bitwise_and(frame, frame, mask=green_mask)
    #
    # # Every color except white
    # low = np.array([0, 42, 0])
    # high = np.array([179, 255, 255])
    # mask = cv2.inRange(hsv_frame, low, high)
    # result = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow(FRAME_WINDOW, frame)
    # cv2.imshow(P1_WINDOW, red)
    # cv2.imshow(P2_WINDOW, blue)
    # # cv2.imshow("Green", green)
    # cv2.imshow(BALL_WINDOW, yellow)
    # # cv2.imshow("Result", result)
    key = cv2.waitKey(1)
    if key == 27:
        break