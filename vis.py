import OpenGL.GL as gl
import numpy as np
import pangolin

from multiprocessing import Process, Queue

class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []
    # self.viewer_init()
    self.state = None
    self.q = None

  def create_viewer(self):
    self.q = Queue()
    self.p = Process(target=self.viewer_thread, args=(self.q,))
    self.p.daemon = True
    self.p.start()

  def viewer_thread(self, q):
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    pangolin.CreateWindowAndBind('Main', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
      pangolin.ModelViewLookAt(0, -10, -8, 
                  0, 0, 0,
                  0, -1, 0))
    self.handler = pangolin.Handler3D(self.scam)

    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
    self.dcam.SetHandler(self.handler)
    # hack to avoid small Pangolin, no idea why it's *2
    self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
    self.dcam.Activate()


  def viewer_refresh(self, q):
    if self.state is None or not q.empty():
      self.state = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)

    if self.state is not None:
      if self.state[0].shape[0] >= 2:
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0][:-1])

      if self.state[0].shape[0] >= 1:
        gl.glColor3f(1.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0][-1:])

      if self.state[1].shape[0] != 0:
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1])

    pangolin.FinishFrame()


  def display(self):
    if self.q is None:
      return
    poses, pts = [], []
    for f in self.frames:
     poses.append(f.pose)
    for p in self.points:
      pts.append(p.pt)
    self.q.put((np.array(poses), np.array(pts)))


class Point(object):
  def __init__(self, m, loc):
    self.frames = []
    self.pt = loc
    self.idx = []

    self.id = len(m.points)
    m.points.append(self)

  def add_observation(self, frame, idx):
    self.frames.append(frame)
    frame.pts[idx] = self
    self.idx.append(idx)
