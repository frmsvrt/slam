import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform as fmt
from skimage.transform import EssentialMatrixTransform
from utils import calculateRt, add_ones, normalize, denormalize
# test statement
# import g2o
np.set_printoptions(suppress=True)

IRt = np.eye(4)

def featureExtractor(img):
  orb = cv2.ORB_create(250)
  feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                 1000,
                 qualityLevel=0.01,
                 minDistance=7)

  # feature detection
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=30) for f in feats]
  kps, des = orb.compute(img, kps)
  return ([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def frame_matches(frame1, frame2):
  # feature matching
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  match = bf.knnMatch(frame1.des, frame2.des, k=2)

  good = []
  idx1, idx2, test = [], [], []
  idxs1, idxs2 = set(), set()
  for m, n in match:
    if m.distance < .80 * n.distance:
      p1 = frame1.kps[m.queryIdx]
      p2 = frame2.kps[m.trainIdx]
      if m.distance < 32:
        if m.queryIdx not in idxs1 and m.trainIdx not in idxs2:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idxs1.add(m.queryIdx)
          idxs2.add(m.trainIdx)
          good.append((p1, p2))

  assert len(good) >= 8

  # feature filtering
  good = np.array(good)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)
  model, mask = ransac((good[:, 0], good[:, 1]),
            EssentialMatrixTransform,
            min_samples=8,
            residual_threshold=.002,
            max_trials=100)
  # TODO: test cv2
  # F, mask = cv2.findFundamentalMat(good[:, 0], good[:, 1], cv2.FM_RANSAC, 0.1, 0.99)
  # E = frame1.K.dot(F).dot(frame1.K)
  return idx1[mask], idx2[mask], calculateRt(model.params)

class Frame(object):
  def __init__(self, m, img, K, pose=np.eye(4)):
    self.frame = img
    self.K = K
    self.kpss, self.des = featureExtractor(self.frame)
    self.pose = np.array(pose)

    self.id = len(m.frames)
    m.frames.append(self)

  @property
  def Kinv(self):
    if not hasattr(self, '_Kinv'):
      self._Kinv = np.linalg.inv(self.K)
    return self._Kinv

  @property
  def kps(self):
    if not hasattr(self, '_kps'):
      self.kpss = np.array(self.kpss)
      self._kps = normalize(self.Kinv, self.kpss)
    return self._kps
