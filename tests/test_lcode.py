import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier


from lcode import LCODE


class TestLCODE(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.classifier = None
        self.detector = None
        np.random.seed(0)

    def tearDown(self) -> None:
        super().tearDown()
        self.classifier = None
        self.detector = None

    def make_testdata(self, n, n_dim, n_class, n_drifts):
        X = np.random.rand(n, n_dim)
        y = np.random.randint(0, n_class, n)
        X[np.arange(n), y] += 1

        data = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
        concepts = np.array_split(data, n_drifts + 1)

        for i in range(len(concepts)):
            concepts[i][:, i % n_dim] *= 2

        data = np.concatenate(concepts, axis=0)
        return data

    def run_detection(self, data, train_size):
        X_train = data[:train_size, :-1]
        y_train = data[:train_size, -1]
        self.classifier.fit(X_train, y_train)
        self.detector.reset(self.classifier, X_train, y_train)

        n = len(data)
        for i in range(train_size, n):
            x = data[i, :-1]
            alert, _ = self.detector.update(x)
            if alert:
                return i, alert
        return n, None

    def test_lcode_0(self):
        '''
        no drift case
        '''
        data = self.make_testdata(n=3000, n_dim=5, n_class=3, n_drifts=0)
        self.classifier = RandomForestClassifier(n_estimators=20, random_state=42)
        self.detector = LCODE()

        n = len(data)
        train_size = n // 10
        head = 0
        alerts = []
        while head < (n - train_size):
            i, alert = self.run_detection(data[head:], train_size)
            head = head + i
            if alert:
                print('Drift is detected at %d.' % head)
                [print('  Class: %d, Feature: %d' % (a[0], a[1])) for a in alert]
                alerts.append((head, alert))

        self.assertEqual(len(alerts), 0)

    def test_lcode_1(self):
        '''
        relevant drift case
        '''
        data = self.make_testdata(n=3000, n_dim=3, n_class=3, n_drifts=2)
        self.classifier = RandomForestClassifier(n_estimators=20, random_state=42)
        self.detector = LCODE()

        n = len(data)
        train_size = n // 10
        head = 0
        alerts = []
        while head < (n - train_size):
            i, alert = self.run_detection(data[head:], train_size)
            head = head + i
            if alert:
                print('Drift is detected at %d.' % head)
                [print('  Class: %d, Feature: %d' % (a[0], a[1])) for a in alert]
                alerts.append((head, alert))

        self.assertEqual(len(alerts), 2)
        self.assertIn(alerts[0][0], range(1000, 1200))
        self.assertIn(alerts[1][0], range(2000, 2200))

    def test_lcode_2(self):
        '''
        including irrelevant drift case
        '''
        data = self.make_testdata(n=5000, n_dim=5, n_class=3, n_drifts=4)
        self.classifier = RandomForestClassifier(n_estimators=20, random_state=42)
        self.detector = LCODE()

        n = len(data)
        train_size = n // 10
        head = 0
        alerts = []
        while head < (n - train_size):
            i, alert = self.run_detection(data[head:], train_size)
            head = head + i
            if alert:
                print('Drift is detected at %d.' % head)
                [print('  Class: %d, Feature: %d' % (a[0], a[1])) for a in alert]
                alerts.append((head, alert))

        self.assertEqual(len(alerts), 2)
        self.assertIn(alerts[0][0], range(1000, 1200))
        self.assertIn(alerts[1][0], range(2000, 2200))


if __name__ == "__main__":
    unittest.main()
