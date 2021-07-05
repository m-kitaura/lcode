# coding: utf-8

from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class DETECTOR_BASE(object, metaclass=ABCMeta):
    def __init__(self):
        self.classifier = None
        self.drifts = None
        self.score = None
        self.train_size = None
        self.data_len = None

    def run(self, data: np.array, train_size: int, batch_size: int = 0, update_policy: bool = True,
            show_progress: bool = False):
        """
        Run drift detection over the data.
        Args:
            data : Data. The last column should be class data.
            train_size : Number of samples to be used to train the classifier.
            batch_size :
                Number of samples to be used for batch-mode drift detection.
                If this value is 0, train_size is used as a batch_size.
            update_policy :
                If True, the classifier will be re-trained when a drift is detected or reference data will be updated
                when a drift isn't detected.
            show_progress :
        """
        alerts = []
        head = 0
        self.data_len = len(data)
        self.train_size = train_size
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = self.train_size

        with tqdm(total=self.data_len, disable=(not show_progress)) as pbar:
            X, y, head, ts = self.make_next_training_set(data, head, train_size)
            if ts == 0:
                return

            self.update_classifier(X, y)
            y_pred = self.classifier.predict(X).tolist()

            pbar.update(len(X))
            while True:
                head_ = head
                if head + 1 > self.data_len:
                    break
                if update_policy and (len(alerts) > 0):
                    alerts = []
                    head, y_pred = self.post_drift(data, head, y_pred)
                else:
                    alerts, head = self.add_sample(data, head)
                pbar.update(head - head_)

            pbar.update(self.data_len - head)

        if head > len(y_pred):
            X = data[len(y_pred):, :-1]
            y_pred = y_pred + self.classifier.predict(X).tolist()

        y_pred = np.array(y_pred)
        self.score = calc_score_over_time(data[:len(y_pred), -1], y_pred, skip=train_size)

        return self.drifts, self.score

    @abstractmethod
    def update_classifier(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def add_sample(self, data, head):
        raise NotImplementedError()

    def post_drift(self, data, head, y_pred):
        X = data[len(y_pred):head, :-1]
        y_pred = y_pred + self.classifier.predict(X).tolist()

        X, y, head, ts = self.make_next_training_set(data, head, self.train_size)
        if ts == 0:
            head = self.data_len
            return head, y_pred
        y_pred = y_pred + self.classifier.predict(X).tolist()

        self.update_classifier(X, y)
        return head, y_pred

    def make_next_training_set(self, data, head, train_size):
        if head + train_size > self.data_len:
            return None, None, self.data_len, 0

        X = data[head:head + train_size, :-1]
        y = data[head:head + train_size, -1]
        head = head + train_size
        ts = train_size

        while len(set(y)) < 2:
            if head + train_size > self.data_len:
                return None, None, self.data_len, 0
            X = np.concatenate((X, data[head:head + train_size, :-1]))
            y = np.concatenate((y, data[head:head + train_size, -1]))
            head = head + train_size
            ts = ts + train_size
        return X, y, head, ts


def calc_score_over_time(y, y_pred, skip, mode='micro'):
    labels = sorted(set(y[skip:].tolist() + y_pred[skip:].tolist()))
    n_labels = len(labels)

    cy = np.tile(y[skip:], [n_labels, 1]).swapaxes(0, 1)
    py = np.tile(y_pred[skip:], [n_labels, 1]).swapaxes(0, 1)

    cp = (cy == labels).astype(float)
    cn = 1 - cp
    pp = (py == labels).astype(float)
    pn = 1 - pp

    if mode == 'micro':
        w = cp.cumsum(axis=0)
    else:
        w = (cp.cumsum(axis=0) > 0).astype(float)
    w = w / np.expand_dims(w.sum(axis=1), 1)

    tp = (cp * pp).cumsum(axis=0)
    fp = (cn * pp).cumsum(axis=0)
    fn = (cp * pn).cumsum(axis=0)
    tn = (cn * pn).cumsum(axis=0)

    cp = cp.cumsum(axis=0)
    cn = cn.cumsum(axis=0)
    pp = pp.cumsum(axis=0)
    pn = pn.cumsum(axis=0)

    precision = np.divide(tp, pp, out=np.zeros_like(tp), where=(pp != 0))
    recall = np.divide(tp, cp, out=np.zeros_like(tp), where=(cp != 0))

    num = 2 * precision * recall
    denom = precision + recall
    f1_score = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
    f1_score = (f1_score * w).sum(axis=1)

    num = tp + tn
    denom = np.ones_like(cp).cumsum(axis=0)
    acc = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
    acc = (acc * w).sum(axis=1)

    num = tp * tn - fp * fn
    denom = cp * cn * pp * pn
    denom[denom <= 0] = 0
    denom = np.sqrt(denom)
    mcc = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
    mcc = (mcc * w).sum(axis=1)

    val = np.array([f1_score, acc, mcc])
    score = pd.DataFrame(val, index=['f1', 'acc', 'mcc'], columns=range(skip, len(y))).T
    return score


def detection_test(detector_cls, data=None, train_size=None):
    np.random.seed(0)

    if data is None:
        n = 5000
        n_dim = 5
        n_class = 3
        n_drifts = 4
        concept_len = n // (n_drifts + 1)

        X = np.random.rand(n, n_class)
        y = np.random.randint(0, n_class, n)
        X[np.arange(n), y] += 1
        X = np.concatenate((X, np.random.rand(n, n_dim - n_class)), axis=1)

        for i in range((n_drifts + 1)):
            X[i * concept_len:(i + 1) * concept_len, i % n_dim] *= 2

        data = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

    if train_size is None:
        train_size = len(data) // 10

    classifier = RandomForestClassifier(n_estimators=20, random_state=42)

    detector = detector_cls(classifier)
    drifts, score = detector.run(data, train_size=train_size)
    print(drifts, score.values[-1])
    return detector
