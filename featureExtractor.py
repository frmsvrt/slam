import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform as fmt
from skimage.transform import EssentialMatrixTransform as emt
from utils import calculateRt, add_ones, normalize, denormalize
# test statement
# import g2o
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

def featureExtractor(img):
        orb = cv2.ORB_create(100)
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                       1000,
                                       qualityLevel=0.01,
                                       minDistance=7)

        # feature detection
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=30) for f in feats]
        kps, des = orb.compute(img, kps)
        # return kps, des
        return ([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def frame_matches(frame1, frame2):
    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    match = bf.knnMatch(frame1.des, frame2.des, k=2)

    good = []
    pose = None
    idx1, idx2, test = [], [], []
    idxs1, idxs2 = set(), set()
    for m, n in match:
        if m.distance < .75 * n.distance:
            p1 = frame1.kps[m.queryIdx]
            p2 = frame2.kps[m.trainIdx]
            if m.distance < 32:
                if m.queryIdx not in idxs1 and m.trainIdx not in idxs2:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idxs1.add(m.queryIdx)
                    idxs2.add(m.trainIdx)
                    good.append((p1, p2))
                    test.append((p1, p2))

    # feature filtering
    good = np.array(good)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    test = np.array(test)
    # data normalization
    good[:, 0, :] = normalize(frame1.Kinv, good[:, 0, :])
    good[:, 1, :] = normalize(frame2.Kinv, good[:, 1, :])

    model, inliers = ransac((good[:, 0], good[:, 1]),
                            emt,
                            min_samples=8,
                            residual_threshold=0.005,
                            max_trials=100)
    pose = calculateRt(model.params)
    return test[inliers], pose


class Frame(object):
    def __init__(self, img, K):
        self.frame = img
        self.K = K
        self.kpss, self.des = featureExtractor(self.frame)

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
