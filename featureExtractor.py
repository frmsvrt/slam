import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform as fmt

class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

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
        if self.last is not None:
            match = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in match:
                if m.distance < .75 * n.distance:
                    p1 = kps[m.queryIdx].pt
                    p2 = self.last['kps'][m.trainIdx].pt
                    good.append((p1, p2))

        if len(good) is not 0:
            good = np.array(good)
            model, inliers = ransac((good[:, 0], good[:, 1]),
                                    fmt,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            good = good[inliers]

        self.last = {'kps' : kps, 'des' : des}
        return good
