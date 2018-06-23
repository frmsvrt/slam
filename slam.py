#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time
from display import Display
from featureExtractor import FeatureExtractor

H = 1920//2
W = 1080//2

F = 740
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0, 1]])

display = Display(W, H)
fe = FeatureExtractor(K)

def process_frame(img):
    img = cv2.resize(img, (H, W))
    match, pose = fe.extract(img)
    if match is None:
        return

    for pt1, pt2 in match:

        # denormalize points 
        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)

        # draw features
        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
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
