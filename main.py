import cv2
import numpy as np
import imutils
import Clor
from deceter import ShapeDetector
from time import sleep
import RPi.GPIO as GPIO
from gpiozero import AngularServo

"""
Sử dụng 4 màu red, blue, green, yellow. => biết được màu của vật thể 
phát hiện các vật thể, đo khoảng cách. Vật thể gồm vuông, tam giác, chữ nhật, tròn. dùng hệ màu BRG => HSV
Xài cùng 1 frame. 
"""

colors = {'red': (0, 0, 255), 'green': (0, 255, 0),
          'blue': (255, 0, 0), 'yellow': (0, 255, 217)}

s = AngularServo(14, min_angle=-90, max_angle=90)
s.angle = -28

# Động cơ DC1
in3 = 24
in4 = 23
enB = 25

# Động cơ DC2
in1 = 2
in2 = 3
enA = 4

GPIO.setmode(GPIO.BCM)

GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(enA, GPIO.OUT)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)

GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(enB, GPIO.OUT)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

p = GPIO.PWM(enA, 1000)
o = GPIO.PWM(enB, 1000)

p.start(30)
o.start(30)


def go_to():
    s.angle = -28
    sleep(1)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)

    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    sleep(2)


def go_back():
    s.angle = -28
    sleep(1)
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)

    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
    sleep(2)


def left():
    s.angle = -90
    sleep(1)

    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)

    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    sleep(2)


def right():
    s.angle = 90
    sleep(1)

    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)

    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    sleep(2)


def stop():
    s.angle = -28
    sleep(2)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  ##Set camera resolution
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        kernel = np.ones((3, 3), np.float32) / 25

        # processing input frame
        frame = imutils.resize(frame, width=400)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        median = cv2.filter2D(frame, -1, kernel)
        # blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # 5,5

        hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
        ratio = frame.shape[0] / float(frame.shape[0])

        red, green, blue, yellow = color_detector.color(hsv)

        color_ = {'red': red, 'green': green, 'blue': blue, 'yellow': yellow}

        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        for key, value in colors.items():
            # kernel = np.ones((9, 9), np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

            mask = cv2.morphologyEx(color_[key], cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.bilateralFilter(mask, 11, 17, 17)  # cancel noise

            cnts = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            center = None
            sd = ShapeDetector()
            # print("Length: {0}".format(len(cnts)))

            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
            # area = cv2.contourArea(cnts)
            # if area > 2000:

            contour_sizes = [(cv2.contourArea(contour), contour)
                             for contour in cnts]
            area_thresh = 0

            area_thresh = 20000

            for c in cnts:
                area = cv2.contourArea(c)
                if area > area_thresh:
                    area = area_thresh
                    big_contour = c

                    M = cv2.moments(big_contour)
                    if M["m00"] == 0:  # this is a line
                        shape = "line"

                    else:
                        cX = int((M["m10"] / M["m00"]) * ratio)
                        cY = int((M["m01"] / M["m00"]) * ratio)
                        shape = sd.detect(big_contour)

                        rect = cv2.minAreaRect(big_contour)
                        frame, dist = distance_detector.get_dist(rect, frame)

                        big_contour = big_contour.astype("float")
                        big_contour *= ratio
                        big_contour = big_contour.astype("int")
                        cv2.drawContours(
                            frame, [big_contour], -1, colors[key], 2)

                        cv2.putText(frame, shape + " " + key, (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[key], 2)
                        # print(type(shape))
                        if key == 'red':
                            print('Stop')
                            cv2.putText(frame, 'Stop', (110 + 100, 50 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[key],
                                        2)
                            stop()
                        elif key == 'green':
                            print('Tien len')
                            cv2.putText(frame, 'Tien len', (110 + 100, 50 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        colors[key], 2)
                            go_to()
                        elif key == 'yellow':
                            print('Di lui')
                            cv2.putText(frame, 'Di lui', (110 + 100, 50 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        colors[key], 2)
                            go_back()
                        elif key == 'blue' and (shape == 'square'):
                            print('Re trai')
                            cv2.putText(frame, 'Re trai', (110 + 100, 50 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        colors[key], 2)
                            left()
                        elif key == 'blue' and (shape == 'circle'):
                            print('Re phai')
                            cv2.putText(frame, 'Re phai', (110 + 100, 50 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        colors[key], 2)
                            right()

                        cv2.imshow('mask', mask)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
