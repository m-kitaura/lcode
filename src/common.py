# coding: utf-8

import json
import hashlib
import h5py
import numpy as np
import pandas as pd


def make_hash(seed, length=8):
    s = {k: float(v) if isinstance(v, int) else v for k, v in seed.items()}
    hs = hashlib.sha1(json.dumps(s, sort_keys=True).encode('utf-8')).hexdigest()
    return hs[:length]


def get_whisker_value(df):
    stats = df.describe()

    iqr = stats.loc['75%'] - stats.loc['25%']
    t_whisker = stats.loc['75%'] + 1.5 * iqr
    b_whisker = stats.loc['25%'] - 1.5 * iqr

    t_whisker = df[df <= t_whisker].max()
    b_whisker = df[df >= b_whisker].min()
    return t_whisker, b_whisker


def make_bin_edges(X, n_bins=30):
    n_features = len(X[0])
    hs, ls = get_whisker_value(pd.DataFrame(X))

    edges = list()
    for i in range(n_features):
        high = hs[i]
        low = ls[i]

        edge = np.linspace(low, high, num=n_bins - 1)
        step = (edge.max() - edge.min()) / (len(edge) - 1)
        edge = np.concatenate([[edge.min() - step], edge, [edge.max() + step]])
        edges.append(edge)
    return np.array(edges)


def make_histograms(X, edges):
    hists = list()
    for i, edge in enumerate(edges):
        val = X[:, i]
        val[val < edge[0]] = edge[0]
        val[val > edge[-1]] = edge[-1]

        hist, _ = np.histogram(val, edge)
        hists.append(hist)
    return np.array(hists)


def hellinger(p, q):
    return np.sqrt(((np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=1)) / np.sqrt(2)


def count_false_alarm(df):
    df['fa1'] = (pd.DataFrame(df['drifts'].values.tolist()).T <= df['n_change_point'].values).sum().T.values
    df['fa2'] = (pd.DataFrame(df['drifts'].values.tolist()).T > df['n_change_point'].values).sum().T.values
    df.loc[df['how_feature'] == 'relevant', 'fa2'] = df.loc[df['how_feature'] == 'relevant', 'fa2'] - 1
    df.loc[df['fa2'] < 0, 'fa2'] = 0
    df['false alarm'] = df['fa1'] + df['fa2']
    df = df.drop(['fa1', 'fa2'], axis=1)
    return df


def load_record(path):
    with h5py.File(path, 'r') as f:
        timestamp = str(f['timestamp'][...])
        data_cond = json.loads(str(f['data_cond'][...]))
        metadata = json.loads(str(f['metadata'][...]))
        clf_cond = json.loads(str(f['clf_cond'][...]))
        detector_cond = json.loads(str(f['detector_cond'][...]))
        drifts = list(f['drifts'][...])
        col = f['score/axis0'][()].astype(str)
        idx = f['score/axis1'][()]
        val = f['score/block0_values'][()]
        score = pd.DataFrame(val, columns=col, index=idx)
    return timestamp, data_cond, clf_cond, metadata, detector_cond, score, drifts


def load_record_lcode(path):
    with h5py.File(path, 'r') as f:
        wd_idx = f['idx'][...]
        wd_val = f['wd'][...]
        th = f['th'][...]
    drift_map = pd.read_hdf(path, key='drift_map')
    return wd_idx, wd_val, th, drift_map


def load_record_md3(path):
    with h5py.File(path, 'r') as f:
        label_requests = list(f['label_requests'][...])
    return label_requests
