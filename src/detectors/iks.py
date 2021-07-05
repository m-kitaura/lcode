# coding: utf-8

import random
from IncrementalKS.IKSSW import IKSSW
from IncrementalKS.IKS import IKS
from detectors.base import DETECTOR_BASE


class IKS_(DETECTOR_BASE):
    def __init__(self, classifier, alpha=0.001):
        super().__init__()
        self.classifier = classifier
        self.alpha = alpha
        self.ca = None
        self.drifts = []
        self.ref = None
        self.target_feat_lst = None
        self.detectors = None
        random.seed(0)

    def update_classifier(self, X, y):
        self.classifier.fit(X, y)
        self.ref = X
        self.target_feat_lst = list(range(X.shape[1]))
        self.detectors = []
        for i in self.target_feat_lst:
            ref = self.ref[:, i]
            self.detectors.append(IKSSW(ref))
        self.ca = IKS.CAForPValue(self.alpha)
    
    def add_sample(self, data, head):
        X = data[head, :-1]
        alerts = []
        for i in self.target_feat_lst:
            self.detectors[i].Increment(X[i])
            if self.detectors[i].Test(self.ca):
                alerts.append(i)

        if len(alerts) > 0:
            self.drifts.append(head)

        return alerts, head + 1


if __name__ == '__main__':
    from detectors.base import detection_test
    detection_test(IKS_)
