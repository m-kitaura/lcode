# coding: utf-8

import os
import json
import h5py
import pandas as pd
import numpy as np

from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier

from common import make_hash, make_bin_edges, make_histograms, hellinger


class DRIFT_INDUCTION():
    def __init__(self, base_dataset_name, path_to_base_dataset, save_dir, train_size=0.1):
        self.clf = RandomForestClassifier(n_estimators=20, random_state=42)
        self.eval_func = matthews_corrcoef
        self.save_dir = save_dir

        self.data = pd.read_csv(path_to_base_dataset).values
        self.train_size = int(train_size * len(self.data))

        self.metadata = {
            'dataset': base_dataset_name,
            'base_path': path_to_base_dataset,
        }

        self.data_shuffled = None
        self.idx_col = None
        self.data_cond = None

    def induct(self, seed, change_point, shuffle_ratio, how_feature, how_class):
        n_instances = len(self.data)
        n_features = len(self.data[0, :-1])
        n_classes = len(set(self.data[:, -1]))
        n_change_point = int(n_instances * change_point)
        self.data_cond = {
            'seed': seed,
            'change_point': change_point,
            'shuffle_ratio': shuffle_ratio,
            'how_feature': how_feature,
            'how_class': how_class,
        }
        if self.exists():
            return

        if 'feature_scores' not in self.metadata:
            feature_scores = self.calc_feature_scores()
        else:
            feature_scores = self.metadata['feature_scores']

        shuffled_classes = DRIFT_INDUCTION.select_shuffled_classes(self.data[:, -1], self.data_cond['how_class'])
        self.idx_col, shuffled_features = self.get_index_col(feature_scores, n_instances, n_change_point,
                                                             **self.data_cond)
        self.data_shuffled = self.shuffle_data(self.data, self.idx_col, shuffled_classes)
        scores, severity, hd = self.check_shuffle_effect(shuffled_classes, n_change_point, feature_scores)

        self.metadata['shuffled_features'] = shuffled_features
        self.metadata['shuffled_classes'] = shuffled_classes
        self.metadata['feature_scores'] = feature_scores
        self.metadata['n_instances'] = n_instances
        self.metadata['n_features'] = n_features
        self.metadata['n_classes'] = n_classes
        self.metadata['n_change_point'] = n_change_point
        self.metadata['train_score'] = scores[0]
        self.metadata['prev_score'] = scores[1]
        self.metadata['post_score'] = scores[2]
        self.metadata['prev_severity'] = severity[0]
        self.metadata['post_severity'] = severity[1]
        self.metadata['prev_hd'] = hd[0]
        self.metadata['post_hd'] = hd[1]

        self.save()

    def exists(self):
        path = self.make_save_file_name()
        return os.path.exists(path)

    def save(self):
        file_path = self.make_save_file_name()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data_cond', data=json.dumps(self.data_cond), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('metadata', data=json.dumps(self.metadata), dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('idx_col', data=json.dumps(self.idx_col), dtype=h5py.special_dtype(vlen=str))

    def calc_feature_scores(self):
        self.clf.fit(self.data[:, :-1], self.data[:, -1])
        return self.clf.feature_importances_.tolist()

    def make_save_file_name(self):
        hs = make_hash(self.data_cond)
        basename = self.metadata['dataset']
        path = os.path.join(self.save_dir, basename, '_'.join([hs, basename + '.hdf5']))
        return path

    def check_shuffle_effect(self, shuffled_classes, n_change_point, feature_scores):
        ref_idx_col = list(self.idx_col)
        ref_idx_col[3] = ref_idx_col[2]     # shuffle order of instances and keep oreder of features
        ref_data = DRIFT_INDUCTION.shuffle_data(self.data, ref_idx_col, shuffled_classes)
        scores = DRIFT_INDUCTION.check_score(self.clf, self.eval_func, ref_data, self.data_shuffled, n_change_point,
                                             self.train_size)
        prev_severity, prev_hd = DRIFT_INDUCTION.check_drift_severity(ref_data[:, :-1], n_change_point, feature_scores)
        post_severity, post_hd = DRIFT_INDUCTION.check_drift_severity(self.data_shuffled[:, :-1], n_change_point,
                                                                      feature_scores)

        return scores, (prev_severity, post_severity), (prev_hd, post_hd)

    @staticmethod
    def select_shuffled_features(scores, shuffle_ratio, how_feature):
        n_shuffle = int(len(scores) * shuffle_ratio)
        rank = np.array(scores).argsort()
        if how_feature == 'relevant':
            rank = rank[::-1]
        rank = rank[:n_shuffle]
        return rank

    @staticmethod
    def select_shuffled_classes(y, how_class, ratio=0.5):
        classes = pd.Series(y).value_counts(normalize=True).sort_index().sort_values().cumsum()
        if how_class == 'major':
            classes = classes[classes > 1 - ratio].index
        elif how_class == 'minor':
            classes = classes[classes <= 1 - ratio].index
        else:
            classes = classes.index
        return sorted(classes.tolist())

    @staticmethod
    def get_index_col(scores, n_instances, n_change_point, seed, shuffle_ratio, how_feature, **kwargs):
        idx = np.arange(n_instances)
        np.random.seed(seed)
        np.random.shuffle(idx)
        idx_prev = idx[:n_change_point].tolist()
        idx_post = idx[n_change_point:].tolist()

        rank = DRIFT_INDUCTION.select_shuffled_features(scores, shuffle_ratio, how_feature)

        cols_prev = np.arange(len(scores))
        cols_post = cols_prev.copy()
        rank_s = np.roll(rank, shift=-1)
        cols_post[rank] = rank_s

        return (idx_prev, idx_post, cols_prev.tolist(), cols_post.tolist()), rank.tolist()

    @staticmethod
    def shuffle_data(data, idx_col, class_change=None):
        idx_prev, idx_post, cols_prev, cols_post = idx_col
        X = data[:, :-1]
        Y = data[:, -1]

        Y_train = Y[idx_prev]
        Y_test = Y[idx_post]

        X_train = X[idx_prev][:, cols_prev]
        if class_change is None:
            X_test = X[idx_post][:, cols_post]
        else:
            X_test = X[idx_post][:, cols_prev]
            X_test_tmp = X[idx_post][:, cols_post]
            if isinstance(class_change, list):
                for c in class_change:
                    X_test[Y_test == c] = X_test_tmp[Y_test == c]
            else:
                X_test[Y_test == class_change] = X_test_tmp[Y_test == class_change]

        X_shuffled = np.concatenate((X_train, X_test), axis=0)
        Y_shuffled = np.concatenate((Y_train, Y_test), axis=0)
        Y_shuffled = np.expand_dims(Y_shuffled, axis=1)
        data_shuffled = np.concatenate((X_shuffled, Y_shuffled), axis=1)

        return data_shuffled

    @staticmethod
    def check_score(clf, eval_func, data, data_shuffled, n_change_point, train_size):
        X = data[:n_change_point, :-1]
        y = data[:n_change_point, -1]
        clf.fit(X[:train_size], y[:train_size])
        train_score = eval_func(y, clf.predict(X))

        X = data[n_change_point:, :-1]
        y = data[n_change_point:, -1]
        prev_score = eval_func(y, clf.predict(X))

        X = data_shuffled[n_change_point:, :-1]
        y = data_shuffled[n_change_point:, -1]
        post_score = eval_func(y, clf.predict(X))

        return train_score, prev_score, post_score

    @staticmethod
    def check_drift_severity(X, n_changepoint, feature_scores):
        edges = make_bin_edges(X)

        hist_prev = make_histograms(X[:n_changepoint], edges)
        hist_post = make_histograms(X[n_changepoint:], edges)
        hist_prev = hist_prev / hist_prev.sum(axis=1)[0]
        hist_post = hist_post / hist_post.sum(axis=1)[0]

        hd = hellinger(hist_prev, hist_post)
        severity = np.dot(feature_scores, hd)
        return severity, hd.tolist()

    @staticmethod
    def load(file_path):
        with h5py.File(file_path, 'r') as f:
            data_cond = json.loads(str(f['data_cond'][...]))
            metadata = json.loads(str(f['metadata'][...]))
            idx_col = json.loads(str(f['idx_col'][...]))

        return data_cond, metadata, idx_col


if __name__ == '__main__':
    pass
