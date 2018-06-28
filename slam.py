#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time
from display import Display
from featureExtractor import Frame, frame_matches
from utils import denormalize

H = 1920//2
W = 1080//2

F = 240
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0, 1]])

display = Display(W, H)
frames = []
def process_frame(img):
    img = cv2.resize(img, (H, W))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return

    # match detection
    matches, pose = frame_matches(frames[-1], frames[-2])

    for (p1, p2) in matches:
        # draw features
        u1, v1 = map(lambda x: int(round(x)), p1)
        u2, v2 = map(lambda x: int(round(x)), p2)
        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.circle(img, (u2, v2), color=(0, 255, 0), radius=3)
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
