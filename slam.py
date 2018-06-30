#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time
from display import Display
from featureExtractor import Frame, frame_matches, IRt
from utils import denormalize
import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue

H = 1920//2
W = 1080//2

F = 240
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0, 1]])


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.viewer_init()
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self, self.q,))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init()
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self):
        if self.state is None or not q.empty():
            self.state = q.get()

        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(ppts)

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(spts)

        pangolin.FinishFrame()


    def display(self):
        poses, pts = [], []
        for f in self.frames:
           poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((poses, pts))


m = Map()
display = Display(W, H)


class Point(object):
    def __init__(self, m, loc):
        self.frames = []
        self.pt = loc
        self.idx = []

        self.id = len(m.points)
        m.points.append(self)

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
    frame = Frame(m, img, K)
    if frame.id == 0:
        return

    f1 = m.frames[-1]
    f2 = m.frames[-2]

    # match detection
    idx1, idx2, Rt = frame_matches(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)
    pts4 = triangulate(f1.pose,
                       f2.pose,
                       f1.kps[idx1],
                       f2.kps[idx2])

    # homofeneus transform
    pts4 /= pts4[:, 3:]

    # points filtering
    _filter = (np.abs(pts4[:, 3]) > .005) & (pts4[:, 2] > 0)

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

    display.draw(img)
    m.display()


if __name__ == "__main__":
    cap = cv2.VideoCapture('./test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
