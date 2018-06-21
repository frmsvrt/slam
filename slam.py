#!/usr/bin/env python3
import cv2
import sys
import time
from display import Display

H = 1920//3
W = 1080//3

display = Display(W, H)

DEBUG = True

def process_frame(img):
    img = cv2.resize(img, (H, W))
    display.draw(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture('./test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
