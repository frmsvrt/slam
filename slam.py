#!/usr/bin/env python3
import cv2
import sys
import time
from display import Display

H = 1920//3
W = 1080//3

display = Display(W, H)
orb = cv2.ORB_create()

class FeatureExtractor(object):
    def __init__(self):
        self.GX = 16//2
        self.GY = 16//2
        self.orb = cv2.ORB_create(100)

    def extract(self, img):
        sy = img.shape[0]//self.GY
        sx = img.shape[1]//self.GX
        points = []
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                img_ = img[ry:ry+sy, rx:rx+sx]
                kp = self.orb.detect(img_, None)
                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    print(p)
                    points.append(p)

        return points

fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (H, W))
    kp = fe.extract(img)

    for p in kp:
        u, v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)

    display.draw(img)



if __name__ == "__main__":
    cap = cv2.VideoCapture('./test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
