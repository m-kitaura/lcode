# coding: utf-8

import os
import io
import json
import h5py
import cProfile
import pstats
import pandas as pd

from experiment import XP_BASE, AGG


class XP_COST(XP_BASE):
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

    def run_detection(self, show_progress):
        update_policy = self.clf_cond['update_policy']
        train_size = self.calc_train_size()
        with cProfile.Profile() as pr:
            self.drifts, self.score = self.detector.run(self.data, train_size, update_policy=update_policy,
                                                        batch_size=1, show_progress=show_progress)

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats('base', 'run')
        time, unit = s.getvalue().splitlines()[0].strip().split(' ')[-2:]
        self.metadata['time'] = time
        self.metadata['unit'] = unit

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, 'w') as f:
            f.create_dataset('metadata', data=json.dumps(self.metadata), dtype=h5py.special_dtype(vlen=str))
        self.score.to_hdf(path, key='score', mode='a')


class AGG_COST(AGG):
    def aggregate_record(self):
        metadatas = list()

        for record in self.records:
            with h5py.File(record, 'r') as f:
                metadata = json.loads(str(f['metadata'][...]))
            metadatas.append(metadata)
        self.results = pd.DataFrame(metadatas)


if __name__ == '__main__':
    pass
