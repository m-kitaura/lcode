# coding: utf-8

from skmultiflow.drift_detection import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN, PageHinkley
from detectors.base import DETECTOR_BASE


class ACC_BASED_DETECTOR(DETECTOR_BASE):
    def __init__(self, classifier, detector_cls, **kwargs):
        super().__init__()
        self.classifier = classifier
        self.detector = detector_cls(**kwargs)
        self.warnings = []
        self.drifts = []

    def update_classifier(self, X, y):
        self.classifier.fit(X, y)
        self.detector.reset()

    def add_sample(self, data, head):
        alerts = []
        X = data[head:, :-1]
        y = data[head:, -1]

        y_ = self.classifier.predict(X)
        error_stream = (y_ != y).astype(int)
        for err in error_stream:
            head += 1
            self.detector.add_element(err)
            drift = self.detector.detected_change()
            warn = self.detector.detected_warning_zone()
            if warn:
                self.warnings.append(head - 1)
            if drift:
                self.drifts.append(head - 1)
                alerts.append(-1)
                break
        return alerts, head


class ADWIN_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, ADWIN, **kwargs)


class DDM_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, DDM, **kwargs)


class EDDM_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, EDDM, **kwargs)


class HDDM_A_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, HDDM_A, **kwargs)


class HDDM_W_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, HDDM_W, **kwargs)


class KSWIN_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, KSWIN, **kwargs)


class PageHinkley_(ACC_BASED_DETECTOR):
    def __init__(self, classifier, **kwargs):
        super().__init__(classifier, PageHinkley, **kwargs)


if __name__ == '__main__':
    from detectors.base import detection_test
    print('ADWIN_')
    detection_test(ADWIN_)

    print('DDM_')
    detection_test(DDM_)

    print('EDDM_')
    detection_test(EDDM_)

    print('HDDM_A_')
    detection_test(HDDM_A_)

    print('HDDM_W_')
    detection_test(HDDM_W_)

    print('KSWIN_')
    detection_test(KSWIN_)

    print('PageHinkley_')
    detection_test(PageHinkley_)
