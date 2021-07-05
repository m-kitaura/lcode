# coding: utf-8

import pickle
import numpy as np

from detectors.base import DETECTOR_BASE


class MD3(DETECTOR_BASE):
    def __init__(self, classifier, theta=2):
        super().__init__()
        self.classifier = classifier
        self.theta = theta
        self.drifts = []
        self.label_requests = []
        np.random.seed(3)

    def update_classifier(self, X, y):
        self.lmd = (self.train_size - 1) / self.train_size
        self.classifier.fit(X, y)
        self.reset_md(X, y)

    def add_sample(self, data, head):
        tail = head + self.batch_size
        if tail > self.data_len:
            tail = self.data_len
        X = data[head:tail, :-1]

        alerts = []
        pp = self.classifier.predict_proba(X)
        ss = MD3.is_in_margin(pp)
        mds = list()
        for s in ss:
            self.MD = self.lmd * self.MD + (1 - self.lmd) * s
            mds.append(self.MD)

        cds = np.where(abs(np.array(mds) - self.MD_ref) > (self.theta * self.sgm_ref))
        if len(cds) == 0 or len(cds[0]) == 0:
            # No drift detected.
            head = tail
            return alerts, head

        head = head + cds[0][0]
        self.label_requests.append(head)
        if self.data_len <= head + self.train_size:
            # Currently drifting but no sufficient sumples.
            head = self.data_len
            return alerts, head

        head = head + self.train_size
        X = data[head - self.train_size:head, :-1]
        y = data[head - self.train_size:head, -1]
        acc = self.classifier.score(X, y)
        if (self.Acc_ref - acc) > (self.theta * self.sgm_acc):
            # Drift is confirmed.
            self.drifts.append(head - 1)
            alerts.append(True)
        self.reset_md(X, y)
        return alerts, head

    def post_drift(self, data, head, y_pred):
        head_ = head - self.train_size
        if head_ > len(y_pred):
            X = data[len(y_pred):head_, :-1]
            y_pred = y_pred + self.classifier.predict(X).tolist()

        X, y, head, ts = self.make_next_training_set(data, head_, self.train_size)
        if ts == 0:
            head = self.data_len
            return head, y_pred
        y_pred = y_pred + self.classifier.predict(X).tolist()

        self.update_classifier(X, y)
        return head, y_pred

    @staticmethod
    def is_in_margin(pp, th=0.5):
        if pp.shape[1] == 1:
            n_in_margin = np.zeros(pp.shape[0])
        else:
            tmp = np.sort(pp, axis=1)
            n_in_margin = (tmp[:, -1] - tmp[:, -2] <= th).astype(int)
        return n_in_margin

    def reset_md(self, X, y):
        self.MD_ref, self.sgm_ref, self.Acc_ref, self.sgm_acc = self.make_ref(X, y)
        self.MD = self.MD_ref
        self.currently_drifting = False

    def make_ref(self, X, y, n_fold=5):
        clf = pickle.dumps(self.classifier)
        n_samples = len(X)
        idx = np.arange(n_samples)
        np.random.shuffle(idx)

        idcs = np.array_split(idx, n_fold)

        accs = list()
        mds = list()
        for i in range(n_fold):
            X_train = np.concatenate([X[idcs[j]] for j in range(n_fold) if j != i])
            y_train = np.concatenate([y[idcs[j]] for j in range(n_fold) if j != i])
            X_test = X[idcs[i]]
            y_test = y[idcs[i]]
            self.classifier.fit(X_train, y_train)
            accs.append(self.classifier.score(X_test, y_test))

            pp = self.classifier.predict_proba(X_test)
            mds.append(MD3.is_in_margin(pp).sum() / len(X_test))

        accs = np.array(accs)
        mds = np.array(mds)
        self.classifier = pickle.loads(clf)
        return mds.mean(), mds.std(), accs.mean(), accs.std()


if __name__ == '__main__':
    from detectors.base import detection_test
    detection_test(MD3)
