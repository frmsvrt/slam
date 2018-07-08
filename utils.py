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

def calcRt(E, first_inliers, second_inliers):
    U, S, Vt = np.linalg.svd(E)
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    # Szelinski 9.19
    R = U.dot(W).dot(Vt)
    t = U[:, 2]

    if not in_front_of_both_frames(first_inliers, second_inliers, R, t):
        t = -U[:, 2]

    if not in_front_of_both_frames(first_inliers, second_inliers, R, t):
        R = U.dot(W.T).dot(Vt)
        t = U[:, 2]

    if not in_front_of_both_frames(first_inliers, second_inliers, R, t):
        t = -U[:, 2]

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    print(Rt)
    return Rt

def in_front_of_both_frames(first_inliers, second_inliers, R, t):
    for first, second in zip(first_inliers, second_inliers):
        first = np.array([first[0], first[1], 1.0])
        second = np.array([second[0], second[0], 1.0])
        first_z = np.dot(R[0, :] - second[0]*R[2, :], t) / np.dot(R[0, :] - second[0]*R[2, :], second)

        first_3d_point = np.array([first[0] * first_z, second[0]*first_z, first_z]).reshape(3,)
        second_3d_points = np.dot(R.T, first_3d_point) - np.dot(R.T, t)
        second_3d_points = second_3d_points.reshape(first_3d_point.shape)
        if first_3d_point[2] < 0 or second_3d_points[0,2] < 0:
            return False
    return True

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
