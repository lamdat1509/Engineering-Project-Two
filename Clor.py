import numpy as np
import cv2

# HSV color space is consists of 3 matrices, 'hue', 'saturation' and 'value'.


def color(input_frame_hsv):
    # Red
    lower_red = np.array([0, 50, 50], np.uint8)
    upper_red = np.array([10, 255, 255], np.uint8)
    mask_red_1 = cv2.inRange(input_frame_hsv, lower_red, upper_red)

    # Red upper mask (170-180)
    lower_red = np.array([170, 50, 50], np.uint8)
    upper_red = np.array([180, 255, 255], np.uint8)
    mask_red_2 = cv2.inRange(input_frame_hsv, lower_red, upper_red)
    mask_red = mask_red_1 + mask_red_2

    # yellow
    lower_yellow = np.array([21, 39, 64], np.uint8)
    upper_yellow = np.array([40, 255, 255], np.uint8)
    mask_yellow = cv2.inRange(input_frame_hsv, lower_yellow, upper_yellow)

    # blue
    lower_blue = np.array([94, 80, 2], np.uint8)
    upper_blue = np.array([179, 255, 255], np.uint8)
    mask_blue = cv2.inRange(input_frame_hsv, lower_blue, upper_blue)

    # green
    lower_green = np.array([41, 39, 64], np.uint8)
    upper_green = np.array([80, 255, 255], np.uint8)
    mask_green = cv2.inRange(input_frame_hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), "uint8")

    red = cv2.dilate(mask_red, kernel)
    yellow = cv2.dilate(mask_yellow, kernel)
    blue = cv2.dilate(mask_blue, kernel)
    green = cv2.dilate(mask_green, kernel)

    return red, green, blue, yellow



