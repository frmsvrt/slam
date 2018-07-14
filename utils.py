import numpy as np
import cv2

def calculateRt(E):
  U, S, Vt = np.linalg.svd(E)
  W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  transl = U[:, 2]
  Rt = np.eye(4)
  Rt[:3, :3] = R
  Rt[:3, 3] = transl
  return Rt

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  return int(round(ret[0])), int(round(ret[1]))

def triangulate(pose1, pose2, pts1, pts2):
  pose1 = np.linalg.inv(pose1)
  pose2 = np.linalg.inv(pose2)
  ret = np.zeros((pts1.shape[0], 4))
  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose2[0]
    A[3] = p[1][1] * pose2[2] - pose2[1]
    _, _, Vt = np.linalg.svd(A)
    ret[i] = Vt[3]

  return ret
