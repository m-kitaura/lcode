# coding: utf-8

import numpy as np

from common import make_bin_edges, make_histograms, hellinger
from detectors.base import DETECTOR_BASE


class HDDDM(DETECTOR_BASE):
    def __init__(self, classifier, gamma=1):
        super().__init__()
        self.classifier = classifier
        self.gamma = gamma
        self.lmd = 0
        self.beta = None
        self.n_bins = None
        self.edges = None
        self.hists_ref = None
        self.hists_cur = None
        self.HD = None
        self.HD_diff = None
        self.drifts = []

    def update_classifier(self, X, y):
        self.classifier.fit(X, y)
        self.reset_ref(X)

    def add_sample(self, data, head):
        alerts = []
        head += self.train_size
        if head > self.data_len:
            head = self.data_len
            return alerts, head

        X = data[head - self.train_size:head, :-1]

        self.hists_cur = make_histograms(X, self.edges)
        self.calc_HD_diff()
        self.update_threshold()

        if (len(self.HD_diff) > 0) and (self.HD_diff[-1] > self.beta):
            self.reset_ref(X)
            self.drifts.append(head - 1)
            alerts.append(True)
        else:
            self.hists_ref = self.hists_ref + self.hists_cur

        return alerts, head

    def post_drift(self, data, head, y_pred):
        X = data[len(y_pred):head, :-1]
        y_pred = y_pred + self.classifier.predict(X).tolist()

        X, y, head, ts = self.make_next_training_set(data, head, self.train_size)
        if ts == 0:
            head = self.data_len
            return head, y_pred
        y_pred = y_pred + self.classifier.predict(X).tolist()

        self.classifier.fit(X, y)
        return head, y_pred

    def reset_ref(self, X):
        self.n_bins = (np.floor(np.sqrt(self.train_size))).astype(int)
        self.edges = make_bin_edges(X, n_bins=self.n_bins)
        self.hists_ref = make_histograms(X, self.edges)
        self.HD = None
        self.HD_diff = list()

    def update_threshold(self):
        if len(self.HD_diff) > 2:
            m = np.mean(self.HD_diff)
            s = np.std(self.HD_diff)
            self.beta = m + self.gamma * s
        else:
            self.beta = np.inf

    def calc_HD_diff(self):
        p = self.hists_ref / self.hists_ref.sum(axis=1)[0]
        q = self.hists_cur / self.hists_cur.sum(axis=1)[0]
        h = hellinger(p, q).mean()
        if self.HD is not None:
            self.HD_diff.append(abs(self.HD - h))
        self.HD = h


if __name__ == '__main__':
    from detectors.base import detection_test
    detection_test(HDDDM)
