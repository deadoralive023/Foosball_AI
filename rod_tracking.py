import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/uonliaquat/Downloads/sample.mp4')

_, frame = cap.read()
h, w, c = frame.shape

while (1):

    _, frame = cap.read()
    frame = frame[220: h - 220, 350: w - 500]
    blur = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 80
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_res = cv2.bitwise_and(frame, frame, mask=white_mask)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([170, 80, 255])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_res = cv2.bitwise_and(frame, frame, mask=black_mask)

    lower_blue = np.array([100, 100, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_res = cv2.bitwise_and(frame, frame, mask=blue_mask)

    lower_red = np.array([150, 15, 0])
    upper_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_res = cv2.bitwise_and(frame, frame, mask=red_mask)

    blue_red_mask = cv2.bitwise_or(blue_mask, red_mask)
    white_blue_red_mask = cv2.bitwise_or(white_mask, blue_red_mask)
    black_white_blue_red_mask = cv2.bitwise_or(black_mask, white_blue_red_mask)

    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(black_white_blue_red_mask, kernel, iterations=2)
    img_dilation = cv2.dilate(black_white_blue_red_mask, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(black_white_blue_red_mask.copy(),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame, contours, -1, (50, 255, 50), 2)

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            # determine the most extreme points along the contour
            rect = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 10, 0), 3)


    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 50:
    #         # extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    #         f_cnt = [0, 0]
    #         for i in range(3):
    #             extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    #             cnt = cnt[cnt != extRight]
    #             f_cnt[0] = f_cnt[0] + extRight[0]
    #             f_cnt[1] = f_cnt[1] + extRight[1]
    #         # extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    #         # extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
    #
    #         f_cnt[0] = int(f_cnt[0] / 3)
    #         f_cnt[1] = int(f_cnt[1] / 3)
    #         cv2.rectangle(frame, (f_cnt[0], f_cnt[1]), (f_cnt[0] + 5, f_cnt[1] + 5), (100, 50, 100), 3)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask white',white_mask)
    # cv2.imshow('res white',white_res)
    # cv2.imshow('mask blue', blue_mask)
    # cv2.imshow('res blue', blue_res)
    # cv2.imshow('mask red', red_mask)
    # cv2.imshow('res red', red_res)
    # cv2.imshow('mask black', black_mask)
    # cv2.imshow('res red', red_res)
    # cv2.imshow('white_blue_red_mask', white_blue_red_mask)
    cv2.imshow('black_white_blue_red_mask', black_white_blue_red_mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
