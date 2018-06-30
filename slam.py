#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time
from display import Display
from featureExtractor import Frame, frame_matches, IRt
from utils import denormalize

H = 1920//2
W = 1080//2

F = 240
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0, 1]])

display = Display(W, H)
frames = []

class Point(object):
    def __init__(self, loc):
        self.frames = []
        self.location = loc
        self.idx = []

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idx.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3],
                                 pose2[:3],
                                 pts1.T,
                                 pts2.T).T

def process_frame(img):
    img = cv2.resize(img, (H, W))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return

    # match detection
    idx1, idx2, Rt = frame_matches(frames[-1], frames[-2])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    pts4 = triangulate(frames[-1].pose,
                       frames[-2].pose,
                       frames[-1].kps[idx1],
                       frames[-2].kps[idx2])

    # homofeneus transform
    pts4 /= pts4[:, 3:]

    # points filtering
    _filter = (np.abs(pts4[:, 3]) > .005) & (pts4[:, 2] > 0)

    for i,p in enumerate(pts4):
        if not _filter[i]:
            continue
        pt = Point(p)
        pt.add_observation(frames[-1], i)
        pt.add_observation(frames[-2], i)

    for (p1, p2) in zip(frames[-1].kps[idx1], frames[-2].kps[idx2]):
        # draw features
        u1, v1 = denormalize(K, p1)
        u2, v2 = denormalize(K, p2)
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
