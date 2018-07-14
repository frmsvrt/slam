#!/usr/bin/env python3
import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import time
from display import Display
from featureExtractor import Frame, frame_matches, IRt
from utils import denormalize, triangulate
from vis import Map, Point

H = 1920//2
W = 1080//2

F = float(os.getenv("F", "800"))
K = np.array([[F, 0, W//2],
       [0, F, H//2],
       [0, 0, 1]])

m = Map()
if os.getenv('d3d') is not None:
  m.create_viewer()
# display = Display(W, H)

def process_frame(img):
  img = cv2.resize(img, (H, W))
  frame = Frame(m, img, K)
  if frame.id == 0:
    return

  f1 = m.frames[-1]
  f2 = m.frames[-2]

  # match detection
  idx1, idx2, Rt = frame_matches(f1, f2)

  # rotation and translation estimation
  f1.pose = np.dot(Rt, f2.pose)

  # points triangulation
  pts4 = triangulate(f1.pose,
           f2.pose,
           f1.kps[idx1],
           f2.kps[idx2])

  # points filtering
  unmatched = np.array([f1.kps[i] is None for i in idx1]).astype(np.bool)
  _filter = (np.abs(pts4[:, 3]) > .005) & (pts4[:, 2] > 0) & unmatched
  # _filter = np.array([f1.kps[i] is None for i in idx1])
  #_filter &= np.abs(pts4[:, 3]) != 0
  pts4 /= pts4[:, 3:]

  for i,p in enumerate(pts4):
    if not _filter[i]:
      continue
    pt = Point(m, p)
    pt.add_observation(f1, i)
    pt.add_observation(f2, i)

  for (p1, p2) in zip(f1.kps[idx1], f2.kps[idx2]):
    # draw features
    u1, v1 = denormalize(K, p1)
    u2, v2 = denormalize(K, p2)
    cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
    cv2.circle(img, (u2, v2), color=(0, 255, 0), radius=3)
    cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

  # display.draw(img)
  if os.getenv('d2d'):
    cv2.imshow('SLAM', img)
    m.display()
    if cv2.waitKey(1) == 27:
     exit(-1)


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage %s <video>" % sys.argv[0])
    exit(-1)
  cap = cv2.VideoCapture(sys.argv[1])

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
