import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform as fmt
from skimage.transform import EssentialMatrixTransform as emt

# test statement
import g2o

np.set_printoptions(suppress=True)

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
    pose = np.concatenate([R, transl.reshape(3, 1)], axis=1)
    return pose

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class FeatureExtractor(object):
    def __init__(self, k):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = k
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))


    def extract(self, img):
        # feature detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                       1000,
                                       qualityLevel=0.01,
                                       minDistance=3)

        # feature detection
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=30) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # feature matching
        good = []
        pose = None
        if self.last is not None:
            match = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in match:
                if m.distance < .75 * n.distance:
                    p1 = kps[m.queryIdx].pt
                    p2 = self.last['kps'][m.trainIdx].pt
                    good.append((p1, p2))

        # feature filtering
        if len(good) is not 0:
            good = np.array(good)

            # data normalization
            good[:, 0, :] = self.normalize(good[:, 0, :])
            good[:, 1, :] = self.normalize(good[:, 1, :])

            model, inliers = ransac((good[:, 0], good[:, 1]),
                                    emt,
                                    min_samples=8,
                                    residual_threshold=0.005,
                                    max_trials=100)
            good = good[inliers]
            pose = calculateRt(model.params)

        self.last = {'kps' : kps, 'des' : des}
        return good, pose
