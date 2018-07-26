import OpenGL.GL as gl
import numpy as np
import pangolin
import g2o

from multiprocessing import Process, Queue

class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []
    # self.viewer_init()
    self.state = None
    self.q = None

  def optimize(self):
    # optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

    for f in self.frames:
      sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[:3, :3], f.pose[:3, 3]))
      sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[2][0], f.K[2][1], 1.0)

      v_se3 = g2o.VertexCam()
      v_se3.set_id(f.id)
      v_se3.set_estimate(sbacam)
      v_se3.set_fixed(f.id == 0)
      opt.add_vertex(v_se3)

    for p in self.points:
      pt = g2o.VertexSBAPointXYZ()
      pt.set_id(p.id + 0x10000)
      pt.set_estimate(p.pt[0:3])
      pt.set_marginalized(True)
      pt.set_fixed(False)
      opt.add_vertex(pt)
      
      for f in p.frames:
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, pt)
        edge.set_vertex(1, opt.vertex(f.id))
        uv = f.kps[f.pts.index(p)]
        edge.set_measurement(uv)

        edge.set_information(np.eye(2))
        edge.set_robust_kernel(robust_kernel)
        opt.add_edge(edge)

    opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(20)

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
    pangolin.CreateWindowAndBind('SLAM', w, h)
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
