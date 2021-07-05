# coding: utf-8

import os
import datetime
import json
import glob
import itertools
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import make_hash, load_record
from drift_induction import DRIFT_INDUCTION


class XP_BASE(object):
    def __init__(self, base_data_dir, drift_info_dir, save_dir, dataset_name, detector_name,
                 detector_cls=None, base_data_type='sorted'):
        self.drift_info_dir = drift_info_dir
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.detector_name = detector_name
        self.detector_cls = detector_cls

        self.data_cond = None
        self.metadata = None
        self.idx_col = None
        self.clf_cond = None
        self.detector_cond = None

        self.base_data_path = os.path.join(base_data_dir, '%s_%s.csv' % (dataset_name, base_data_type))
        self.data = None
        self.clf = None
        self.detector = None
        self.drifts = None
        self.score = None

        self.set_default()

    def set_default(self):
        # data conditions
        self.data_cond = {
            'seed': 0,
            'change_point': 0.5,
            'shuffle_ratio': 0.2,
            'how_feature': 'relevant',
            'how_class': 'all',
        }

        # classifier conditions
        self.clf_cond = {
            'n_estimators': 20,
            'random_state': 42,
            'update_policy': True,
            'train_size': 0.1,
        }

        self.detector_cond = {}

    def make_save_file_name(self):
        hs = make_hash(self.data_cond)
        hs2 = make_hash(self.detector_cond)
        path = os.path.join(self.save_dir, self.dataset_name, self.detector_name,
                            '_'.join([hs, hs2, self.dataset_name + '.hdf5']))
        return path

    def make_data(self):
        hs = make_hash(self.data_cond)
        path = os.path.join(self.drift_info_dir, self.dataset_name, hs + '_' + self.dataset_name + '.hdf5')
        if not os.path.exists(path):
            print(self.data_cond, path)
            raise Exception('Condition mismatch!')

        _, self.metadata, self.idx_col = DRIFT_INDUCTION.load(path)
        base_data = pd.read_csv(self.base_data_path).values
        self.data = DRIFT_INDUCTION.shuffle_data(base_data, self.idx_col, self.metadata['shuffled_classes'])

    def calc_train_size(self):
        if self.clf_cond['train_size'] < 1:
            train_size = int(self.clf_cond['train_size'] * len(self.data))
        else:
            train_size = self.clf_cond['train_size']
        return train_size

    def init_clf(self):
        n_estimators = self.clf_cond['n_estimators']
        random_state = self.clf_cond['random_state']
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def init_detector(self):
        if self.detector_cls is not None:
            self.detector = self.detector_cls(self.clf, **self.detector_cond)

    def run_detection(self, show_progress):
        update_policy = self.clf_cond['update_policy']
        train_size = self.calc_train_size()
        self.drifts, self.score = self.detector.run(self.data, train_size, update_policy=update_policy,
                                                    show_progress=show_progress)

    def run_experiment(self, data_prms=None, detector_prms=None, show_progress=False):
        if isinstance(data_prms, dict):
            itr = itertools.product(*data_prms.values())
            data_cond_df = pd.DataFrame(list(itr), columns=data_prms.keys())
        else:
            data_cond_df = pd.DataFrame([self.data_cond])
        if isinstance(detector_prms, dict):
            itr = itertools.product(*detector_prms.values())
            detector_cond_df = pd.DataFrame(list(itr), columns=detector_prms.keys())
        else:
            detector_cond_df = pd.DataFrame([self.detector_cond])

        for _, data_cond in data_cond_df.iterrows():
            for k, v in data_cond.items():
                self.data_cond[k] = v

            for _, detector_cond in detector_cond_df.iterrows():
                for k, v in detector_cond.items():
                    self.detector_cond[k] = v

                path = self.make_save_file_name()
                if os.path.exists(path):
                    continue

                self.make_data()
                self.init_clf()
                self.init_detector()
                self.run_detection(show_progress)
                self.save(path)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime('%y%m%d_%H%M%S')
        with h5py.File(path, 'w') as f:
            f.create_dataset('timestamp', data=timestamp, dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('data_cond', data=json.dumps(self.data_cond), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('clf_cond', data=json.dumps(self.clf_cond), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('metadata', data=json.dumps(self.metadata), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('detector_cond', data=json.dumps(self.detector_cond), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('drifts', data=self.drifts)

        self.score.to_hdf(path, key='score', mode='a')


class XP_LCODE(XP_BASE):
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


class XP_MD3(XP_BASE):
    def save(self, path):
        super().save(path)
        with h5py.File(path, 'a') as f:
            f.create_dataset('label_requests', data=self.detector.label_requests)


class AGG(object):
    def __init__(self, record_dir, dataset_name, detector_name):
        self.record_dir = record_dir
        self.dataset_name = dataset_name
        self.detector_name = detector_name
        save_path = os.path.join(self.record_dir, self.dataset_name, self.detector_name)
        self.records = glob.glob(os.path.join(save_path, '*.hdf5'))
        self.results = None
        self.aggregate_record()

    def aggregate_record(self):
        timestamps = list()
        data_conds = list()
        clf_conds = list()
        metadatas = list()
        scores = list()
        detector_conds = list()
        driftss = list()
        index = list()

        for record in self.records:
            timestamp, data_cond, clf_cond, metadata, detector_cond, score, drifts = load_record(record)
            timestamps.append(timestamp)
            data_conds.append(data_cond)
            clf_conds.append(clf_cond)
            metadatas.append(metadata)
            detector_conds.append(detector_cond)
            scores.append(score)
            driftss.append(drifts)
            index.append('_'.join([make_hash(data_cond), make_hash(detector_cond)]))

        data_conds = pd.DataFrame(data_conds, index=index)
        clf_conds = pd.DataFrame(clf_conds, index=index)
        metadatas = pd.DataFrame(metadatas, index=index)
        detector_conds = pd.DataFrame(detector_conds, index=index)
        driftss = pd.Series(driftss, index=index, name='drifts')
        oa_scores = pd.DataFrame(columns=scores[0].columns, index=index)
        val = list()
        for score in scores:
            val.append(score.iloc[-1].values.tolist())
        oa_scores.loc[:, :] = np.array(val)
        records = pd.Series(self.records, index=index, name='record')

        results = pd.concat([data_conds, clf_conds, metadatas, detector_conds, driftss, oa_scores, records], axis=1)
        results['num_of_detection'] = [len(drift) for drift in driftss]
        results = self.list_to_tuple(results)
        self.results = results

    @staticmethod
    def list_to_tuple(df):
        cols = df.columns[df.dtypes == object]
        for col in cols:
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        return df

    def select_results(self, ex_keys=[], **kwargs):
        df = self.results.copy()
        for key in kwargs:
            if key not in ex_keys:
                df = df[df[key] == kwargs[key]]
        return df


if __name__ == '__main__':
    pass
