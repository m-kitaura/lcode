# coding: utf-8

import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from detectors.base import DETECTOR_BASE


class MWU(DETECTOR_BASE):
    def __init__(self, classifier, test=mannwhitneyu, alpha=0.01, method='hs'):
        super().__init__()
        self.classifier = classifier
        self.test = test
        self.alpha = alpha
        self.method = method
        self.drifts = []
        self.ref = None
        self.target_feat_lst = None
        self.step_size = None

    def update_classifier(self, X, y):
        self.classifier.fit(X, y)
        self.ref = X
        self.target_feat_lst = list(range(X.shape[1]))
        self.step_size = 1

    def add_sample(self, data, head):
        alerts = []
        head += self.step_size
        if head > self.data_len:
            head = self.data_len
            return alerts, head

        X = data[head - self.train_size:head, :-1]
        ps = []
        for i in self.target_feat_lst:
            ref = self.ref[:, i]
            cur = X[:, i]
            if np.all(ref == cur):
                p = 1
            else:
                _, p = self.test(ref, cur, alternative='two-sided')
            ps.append(p)
        reject, _, _, _ = multipletests(ps, alpha=self.alpha, method=self.method)
        alerts = [i for i, r in zip(self.target_feat_lst, reject) if r]

        if reject.any():
            self.drifts.append(head - 1)

        return alerts, head


if __name__ == '__main__':
    from detectors.base import detection_test
    detection_test(MWU)
