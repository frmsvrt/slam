#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time
from display import Display
from featureExtractor import FeatureExtractor

H = 1920//3
W = 1080//3

display = Display(W, H)
fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (H, W))
    match = fe.extract(img)
    if match is None:
        return

    for pt1, pt2 in match:
        u, v = map(lambda x: int(round(x)), pt1)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    display.draw(img)



if __name__ == "__main__":
    cap = cv2.VideoCapture('./test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
