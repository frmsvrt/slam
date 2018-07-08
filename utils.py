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
    print(Rt)
    return Rt

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    return int(round(ret[0])), int(round(ret[1]))

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3],
                                 pose2[:3],
                                 pts1.T,
                                 pts2.T).T
