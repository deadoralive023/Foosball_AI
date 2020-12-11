import cv2
import numpy as np

cap = cv2.VideoCapture('sample.mp4')

_, frame = cap.read()
h, w, c = frame.shape

table_params = [140, h - 140, 200, w - 340]
rod_roi_params = [300, h - 300, 210, w - 350]

while True:

    _, frame = cap.read()
    result = frame
    result = result[table_params[0]: table_params[1], table_params[2]: table_params[3]]
    # frame = frame[220: h - 220, 230: w - 370]
    frame = frame[rod_roi_params[0]: rod_roi_params[1], rod_roi_params[2]:rod_roi_params[3]]
    blur = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 80
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_res = cv2.bitwise_and(blur, blur, mask=white_mask)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([170, 80, 255])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_res = cv2.bitwise_and(blur, blur, mask=black_mask)

    lower_blue = np.array([100, 100, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_res = cv2.bitwise_and(blur, blur, mask=blue_mask)

    lower_red = np.array([150, 15, 0])
    upper_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_res = cv2.bitwise_and(blur, blur, mask=red_mask)

    blue_red_mask = cv2.bitwise_or(blue_mask, red_mask)
    white_blue_red_mask = cv2.bitwise_or(white_mask, blue_red_mask)
    black_white_blue_red_mask = cv2.bitwise_or(black_mask, white_blue_red_mask)

    kernel = np.ones((3, 3), np.uint8)
    black_white_blue_red_mask = cv2.erode(black_white_blue_red_mask, kernel, iterations=1)
    black_white_blue_red_mask = cv2.dilate(black_white_blue_red_mask, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(black_white_blue_red_mask.copy(),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame, contours, -1, (50, 255, 50), 2)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            # determine the most extreme points along the contour
            rect = cv2.boundingRect(cnt)
            cv2.rectangle(result, (rod_roi_params[2] - table_params[2] + rect[0], 0), (rod_roi_params[2] - table_params[2] + rect[0] + rect[2], result.shape[0]), (0, 10, 0), 3)
            # cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 10, 0), 3)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
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
    cv2.imshow('mask white',white_mask)
    # cv2.imshow('res white',white_res)
    cv2.imshow('mask blue', blue_mask)
    # cv2.imshow('res blue', blue_res)
    cv2.imshow('mask red', red_mask)
    # cv2.imshow('res red', red_res)
    cv2.imshow('mask black', black_mask)
    # cv2.imshow('res red', red_res)
    cv2.imshow('white_blue_red_mask', white_blue_red_mask)
    cv2.imshow('black_white_blue_red_mask', black_white_blue_red_mask)
    cv2.imshow('result', result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
