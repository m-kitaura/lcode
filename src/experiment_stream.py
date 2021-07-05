# coding: utf-8

import os
import h5py
import pandas as pd

from experiment import XP_BASE


class XPS_BASE(XP_BASE):
    def __init__(self, base_data_dir, save_dir, dataset_name, detector_name, detector_cls=None, base_data_type='cat'):
        super().__init__(base_data_dir, None, save_dir, dataset_name, detector_name,
                         detector_cls=detector_cls, base_data_type=base_data_type)

        self.metadata = {
            'dataset': dataset_name,
            'base_path': os.path.join(base_data_dir, '%s_%s.csv' % (dataset_name, base_data_type)),
        }

    def set_default(self):
        self.data_cond = {
            'type': 'stream'
        }

        # classifier conditions
        self.clf_cond = {
            'n_estimators': 20,
            'random_state': 42,
            'update_policy': True,
            'train_size': 1000,
        }

        self.detector_cond = {}

    def make_data(self):
        self.data = pd.read_csv(self.base_data_path).values


class XPS_LCODE(XPS_BASE):
    def save(self, path):
        idx, wd, drift_map = self.detector._get_results()
        self.drifts = list(sorted(set(drift_map['position'].values.tolist())))
        super().save(path)
        with h5py.File(path, 'a') as f:
            f.create_dataset('idx', data=idx)
            f.create_dataset('wd', data=wd)
            f.create_dataset('th', data=self.detector.l1_th)
        drift_map.to_hdf(path, key='drift_map', mode='a')
        pd.DataFrame(self.detector.l2_log).to_hdf(path, key='l2_log', mode='a')


class XPS_MD3(XPS_BASE):
    def save(self, path):
        super().save(path)
        with h5py.File(path, 'a') as f:
            f.create_dataset('label_requests', data=self.detector.label_requests)


class XPS_UPDATE(XPS_BASE):
    def run_detection(self, show_progress):
        update_policy = self.clf_cond['update_policy']
        train_size = self.calc_train_size()
        change_points = list(range(train_size, len(self.data), train_size))
        self.drifts, self.score = self.detector.run(self.data, train_size, update_policy=update_policy,
                                                    change_points=change_points, show_progress=show_progress)


if __name__ == '__main__':
    pass
