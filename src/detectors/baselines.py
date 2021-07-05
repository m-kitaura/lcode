# coding: utf-8

import numpy as np
from detectors.base import calc_score_over_time


class W_UPDATE(object):
    def __init__(self, classifier, **kwargs):
        self.clf = classifier
        self.drifts = []

    def run(self, data, train_size, update_policy=None, change_points=None, show_progress=None):
        self.data_len = len(data)
        if change_points is None:
            change_points = [self.data_len // 2]

        head = 0
        X = data[head:head + train_size, :-1]
        y = data[head:head + train_size, -1]
        self.clf.fit(X, y)
        y_pred = self.clf.predict(X).tolist()
        head = train_size

        for cp in change_points:
            if (cp + train_size) > self.data_len:
                break
            if (cp + train_size) <= head:
                continue
            self.drifts.append(cp)

            X = data[head:cp + train_size, :-1]
            y_pred = y_pred + self.clf.predict(X).tolist()

            X = data[cp:cp + train_size, :-1]
            y = data[cp:cp + train_size, -1]
            self.clf.fit(X, y)
            head = cp + train_size

        if head < self.data_len:
            X = data[head:, :-1]
            y_pred = y_pred + self.clf.predict(X).tolist()

        y_pred = np.array(y_pred)
        score = calc_score_over_time(data[:, -1], y_pred, skip=train_size)
        return self.drifts, score


class WO_UPDATE(object):
    def __init__(self, classifier, **kwargs):
        self.clf = classifier

    def run(self, data, train_size, update_policy=None, show_progress=None):
        drifts = []
        X = data[:train_size, :-1]
        y = data[:train_size, -1]
        self.clf.fit(X, y)
        X = data[:, :-1]
        y = data[:, -1]
        y_pred = self.clf.predict(X)
        score = calc_score_over_time(data[:, -1], y_pred, skip=train_size)
        return drifts, score


class MA1(object):
    def __init__(self, classifier, **kwargs):
        super().__init__()

    def run(self, data, train_size, update_policy=None):
        self.drifts = []
        self.score = calc_score_over_time(data[:, -1], np.array([0] + list(data[:-1, -1])), skip=train_size)
        return self.drifts, self.score


if __name__ == '__main__':
    from detectors.base import detection_test
    print('update')
    detection_test(W_UPDATE)

    print('no update')
    detection_test(WO_UPDATE)

    print('1 moving average')
    detection_test(MA1)
