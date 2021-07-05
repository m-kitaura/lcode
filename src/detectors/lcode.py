# coding: utf-8

import warnings
import pandas as pd
import numpy as np
import shap

from detectors.base import DETECTOR_BASE


class LCODE(DETECTOR_BASE):
    def __init__(self, classifier, alpha=2, gamma=0.002, categorical_feat=[], include=0.99, exclude=[], verbosity=0):
        """
        Initialize LCODE Class.
        Args:
            classifier (Object): A tree classifier object. This classifier should be supported by shap explainer.
            alpha (float, optional): Parameter to make the threshold value of drift detection. Defaults to 2.
            gamma (float, optional): Parameter of the smoothing factor of shap distributions. Defaults to 0.002.
            categorical_feat (list, optional): List of index of categorical features. Defaults to [].
            include (float or list, optional): If this value is float, this value is used as a threshold to determin
                relevant feature automatically. If this value is a list, drift detection runs based on only the features
                included in the list. Defaults to 0.99.
            exclude (list, optional): List of index of features that should be excluded from drift detection. Defaults
                to [].
            verbosity (int, optional): Verbosity level of reports. Defaults to 1.
        """
        super().__init__()

        # variables to record detected drifts
        self.sample_counts = 0
        self.drifts = {}

        # LCODE parameters
        self.classifier = classifier
        self.alpha = alpha
        self.gamma = gamma
        self.categorical_feat = categorical_feat
        self.include = include
        self.exclude = exclude
        self.verbosity = verbosity
        self.n_bin_max = 50
        self.th_min = 0.01 * 0.01

        # variables for drift detection
        self.n_bin = None
        self.explainer = None
        self.n_feature = None
        self.target_feat_list = None
        self.n_tgt_feat = None
        self.class_list = None
        self.n_class = None
        self.feature_bin_edges = None
        self.shap_bin_edges = None
        self.shap_step = None
        self.shap_bias = None
        self.sample_counts_ad = None
        self.wd_mean = None
        self.wd_var = None
        self.th = None
        self.ref_dist = None
        self.obs_shap_pdf = None
        self.exp_shap_pdf = None

        # variables for visualization
        self.l1_idx = []
        self.l1_wd = []
        self.l1_th = []
        self.l1_class_list = {}
        self.l2_log = []

        warnings.filterwarnings("ignore", message="Setting feature_perturbation = \"tree_path_dependent\" because no "
                                "background data was given.")

    def update_classifier(self, X, y):
        """
        Update classifier with given data 'X' and label 'y'.
        Binning edges and reference joint frequency tables will be also updated.
        This method should be called after a drift is detected.
        Args:
            X (DataFrame): train data
            y (DataFrame): train label
        """
        self.n_feature = X.shape[-1]
        self.sample_counts += len(X)

        self.classifier.fit(X, y)
        self.explainer = shap.TreeExplainer(self.classifier)
        shap_values, self.shap_bias = self._get_shap(X)
        self.target_feat_list = self._select_relevant_features(shap_values)
        self.n_tgt_feat = len(self.target_feat_list)
        self.feature_bin_edges = self._make_feature_bin_edges(X)
        self.shap_bin_edges, self.shap_step = self._make_shap_bin_edges(shap_values)

        self.class_list = np.array(sorted(set(y))).astype(int).tolist()
        self.n_class = len(self.class_list)
        self.l1_class_list[self.sample_counts] = self.class_list

        self.sample_counts_ad = 0
        self.wd_mean = np.zeros((self.n_tgt_feat, self.n_class))
        self.wd_var = np.zeros((self.n_tgt_feat, self.n_class))
        self.th = np.infty

        self.ref_dist = self._make_joint_histograms(X, shap_values)
        self.obs_shap_pdf = np.zeros((self.n_tgt_feat, self.n_class, self.n_bin_max))
        self.exp_shap_pdf = np.zeros((self.n_tgt_feat, self.n_class, self.n_bin_max))

    def add_sample(self, data, head):
        """
        Inspect a drift based on the added sample.
        Args:
            data: A stream data.
            head: Position of the sample to be processed.
        Returns:
            alert: A list of drifted feature ids.
            head: Position of the next sample.
        """
        tail = head + self.batch_size
        if tail > self.data_len:
            tail = self.data_len
        X = data[head:tail, :-1]
        shap_values, _ = self._get_shap(X)
        scs = self.sample_counts + np.arange(len(X)) + 1
        scads = self.sample_counts_ad + np.arange(len(X)) + 1

        ws = 1 / scads
        ws[ws < self.gamma] = self.gamma

        x_tgts = X[:, self.target_feat_list]
        shap_tgts = shap_values[:, self.target_feat_list]
        alert = []
        for sc, scad, w, x_tgt, shap_tgt in zip(scs, scads, ws, x_tgts, shap_tgts):
            head += 1
            self.sample_counts = sc
            self.sample_counts_ad = scad
            self._update_distributions(w, x_tgt, shap_tgt)
            wd = self._calc_distance(self.obs_shap_pdf, self.exp_shap_pdf)
            self._update_threshold(wd)
            alert = self._inspect_drift(wd)

            self._save_l1_log(wd)
            self._save_l2_log(alert)

            if len(alert) > 0:
                break

        return alert, head

    # ==========================================================================
    # Internal methods
    def _update_distributions(self, w, X, shap_values):
        f, c = np.indices((self.n_tgt_feat, self.n_class))
        pos = (self.shap_bin_edges < np.expand_dims(shap_values, axis=-1)).sum(axis=-1) - 1
        pos = np.minimum(np.maximum(0, pos), len(self.shap_bin_edges) - 2)

        self.obs_shap_pdf *= (1 - w)
        self.obs_shap_pdf[f.flatten(), c.flatten(), pos.flatten()] += w

        xq = (self.feature_bin_edges < np.expand_dims(X, axis=-1)).sum(axis=-1) - 1
        xq = np.minimum(np.maximum(0, xq), (~np.isnan(self.feature_bin_edges)).sum(axis=1) - 2)
        exp_dist = self.ref_dist[np.arange(len(xq)), :, xq]
        denom = exp_dist.sum(axis=-1)
        denom = np.tile(denom, [exp_dist.shape[-1], 1, 1]).swapaxes(0, 1).swapaxes(1, 2)
        exp_dist = np.divide(exp_dist, denom, out=self.exp_shap_pdf.copy(), where=(denom != 0))

        self.exp_shap_pdf *= (1 - w)
        self.exp_shap_pdf += (w * exp_dist)

    def _calc_distance(self, p: np.array, q: np.array) -> np.array:
        # 1-Wasserstein distance
        d = np.abs(p.cumsum(axis=-1) - q.cumsum(axis=-1)).mean(axis=-1)
        return d * self.shap_step

    def _inspect_drift(self, wd):
        alert = np.array(np.where(self.th < wd)).T
        if len(alert):
            alert = [(self.target_feat_list[i], self.class_list[j]) for (i, j) in alert]
            self.drifts[self.sample_counts] = alert
        return alert

    def _update_threshold(self, wd):
        if self.sample_counts_ad > np.sqrt(self.train_size):
            # update threshold value only after enough number of samples are passed
            ths = self.wd_mean + self.alpha * np.sqrt(self.wd_var)
            self.th = ths.max()
            self.th = max(self.th, self.th_min)
        self._update_wd_mean_var(wd)

    def _update_wd_mean_var(self, wd):
        n = self.sample_counts_ad
        mp = self.wd_mean
        vp = self.wd_var
        m = ((n - 1) * mp + wd) / n
        v = ((n - 1) * (vp + mp**2) + wd**2) / n - m**2
        v[v < 0] = 0

        self.wd_mean = m
        self.wd_var = v

    def _get_shap(self, X):
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        shap_values = np.array(shap_values)
        if len(X.shape) == 1:
            shap_values = shap_values.swapaxes(0, 1)
        else:
            shap_values = shap_values.swapaxes(0, 1).swapaxes(1, 2)
        shap_bias = self.explainer.expected_value
        return shap_values, shap_bias

    def _select_relevant_features(self, shap_values):
        """
        Select relevant features based on settings ('include' and 'exclude').
        Args:
            shap_values (DataFrame): SHAP values used in automatic feature selection.
        Returns:
            list: A list of selected features.
        """
        # if type of include is a list, only consider one or several selected features
        if isinstance(self.include, list):
            feat_lst = self.include
        else:
            if self.include < 1:
                feat_lst = self._auto_feat_selection(shap_values, self.include)
            else:
                feat_lst = shap_values.columns.tolist()

            # exclude the features that we will not consider based on the prior knowledge
            feat_lst = [i for i in feat_lst if i not in self.exclude]
        return sorted(feat_lst)

    def _auto_feat_selection(self, shap_values, threshold):
        """
        select important features based on the global shapley values
        parameters:
            shap_values: pandas dataframe, shapley values in a time window
            threshold: value in [0,1], the threshold for percentage of contribution to include
        return:
            feat_lst: list, index of selected features
        """
        # calculate global shap, which is the mean(|SHAP value|) of each feature
        global_shap_ratio = np.abs(shap_values).mean(axis=(0, 2))
        global_shap_ratio = global_shap_ratio / global_shap_ratio.sum()
        cumulative_contribution = pd.Series(global_shap_ratio).sort_values(ascending=False).cumsum()
        feat_lst = cumulative_contribution[cumulative_contribution.shift(1, fill_value=0) <= threshold].index.tolist()
        return sorted(feat_lst)

    def _make_shap_bin_edges(self, shap_values):
        shap_min = shap_values.min()
        shap_max = shap_values.max()
        shap_bin_edges = np.linspace(shap_min, shap_max, self.n_bin_max - 1)
        step = shap_bin_edges[1] - shap_bin_edges[0]
        shap_bin_edges = np.array([shap_bin_edges[0] - step, *shap_bin_edges, shap_bin_edges[-1] + step])
        return shap_bin_edges, step

    def _make_feature_bin_edges(self, X):
        """
        Make bin edges based on a number of unique values of each features.
        Each edges has lower and upper out-of-range bins for unobserved values in training data.
        Args:
            X (DataFrame): data to make bin edges.
        Returns:
            list: A list of feature bin edges.
        """
        # we set the number of bins for continuous variables as "|lfloor |sqrt (w) |rfloor"
        self.n_bin = int(np.sqrt(len(X)))
        self.n_bin = min(self.n_bin, self.n_bin_max)

        bin_edges = []
        n_bin_max = 0
        for feature_id in self.target_feat_list:
            unique_values = np.unique(X[:, feature_id])
            if (len(unique_values) <= self.n_bin) or (feature_id in self.categorical_feat):
                # categorical features or discrete values less than n_bin
                bins = np.array(sorted(unique_values))
                if len(bins) == 1:
                    bins = np.array([bins[0] - 1, bins[0], bins[0] + 1])
                else:
                    bins = np.array([2 * bins[0] - bins[1]] + bins.tolist() + [2 * bins[-1] - bins[-2]])
                    bins = (bins[:-1] + bins[1:]) / 2
            else:
                # features with continuos values or discrete values larger than n_bin (frequency based bining)
                # to add two out-of-range bins, apply qcut with number of bins = (n_bin - 2)
                _, bins = pd.qcut(X[:, feature_id], q=(self.n_bin - 2), labels=False, retbins=True, duplicates='drop')
                bins = np.array([2 * bins[0] - bins[1]] + bins.tolist() + [2 * bins[-1] - bins[-2]])
                # pd.qcut retruns left-open bins but we use this in np.histogram2d which uses right-open bins so we add
                # small value to keep consistency.
                bins += 1e-10

            bin_edges.append(bins)
            n_bin_max = max(n_bin_max, len(bins))

        fbe = np.full((self.n_tgt_feat, n_bin_max), np.nan, dtype=float)
        for i, bins in enumerate(bin_edges):
            fbe[i, :len(bins)] = bins

        return fbe

    def _make_joint_histograms(self, X: np.array, shap_values: np.array) -> np.array:
        jhss = []

        X_ = X[:, self.target_feat_list]
        sv = shap_values[:, self.target_feat_list]

        xmin = np.nanmin(self.feature_bin_edges, axis=1)
        xmax = np.nanmax(self.feature_bin_edges, axis=1)
        X_ = np.clip(X_, xmin, xmax)
        sv = np.clip(sv, self.shap_bin_edges[0], self.shap_bin_edges[-1])
        for i, _ in enumerate(self.target_feat_list):
            jhs = []
            xi = X_[:, i]
            fbe = self.feature_bin_edges[i]
            for c in range(self.n_class):
                jh, _, _ = np.histogram2d(xi, sv[:, i, c], (fbe, self.shap_bin_edges))
                jhs.append(jh)
            jhss.append(np.array(jhs))
        return np.array(jhss)

    # ==========================================================================
    # utility methods
    def _get_results(self):
        idx = []
        wd = []
        drift_map = [(dp, df[0], df[1]) for dp, dfs in self.drifts.items() for df in dfs]
        drift_map = pd.DataFrame(drift_map, columns=['position', 'feature', 'class'])

        if self.verbosity > 0:
            idx = np.array(self.l1_idx)
            classes = sorted(list(set(np.concatenate(list(self.l1_class_list.values())))))
            splits = [(idx < t).sum() for t in self.l1_class_list.keys()]
            wds = np.split(np.array(self.l1_wd, dtype='object'), splits)[1:]

            for i, w in enumerate(wds):
                cls_idx = list(self.l1_class_list.values())[i]
                cls_idx = [classes.index(j) for j in cls_idx]
                tmp_wd = np.full((w.shape[0], w.shape[1], len(classes)), np.nan)
                tmp_wd[:, :, cls_idx] = np.array(w.tolist())
                wd.append(tmp_wd)
            wd = np.concatenate(wd)

        return idx, wd, drift_map

    def _save_l1_log(self, wd):
        if self.verbosity < 1:
            return

        self.l1_idx.append(self.sample_counts)
        self.l1_th.append(self.th)
        wd_tmp = np.full((self.n_feature, self.n_class), np.nan)
        wd_tmp[self.target_feat_list] = wd
        self.l1_wd.append(wd_tmp.tolist())

    def _save_l2_log(self, alert):
        if self.verbosity < 2:
            return

        for (f_, c_) in alert:
            f = self.target_feat_list.index(f_)
            c = self.class_list.index(c_)
            self.l2_log.append(
                {
                    'position': self.sample_counts,
                    'feature': f_,
                    'class': c_,
                    'sbe': self.shap_bin_edges,
                    'fbe': self.feature_bin_edges[f],
                    'ref': self.ref_dist[f, c],
                    'esp': self.exp_shap_pdf[f, c],
                    'osp': self.obs_shap_pdf[f, c],
                }
            )


class LCODE_WD(LCODE):
    def _calc_distance(self, p: np.array, q: np.array) -> np.array:
        # 1-Wasserstein distance
        d = np.abs(p.cumsum(axis=-1) - q.cumsum(axis=-1)).mean(axis=-1)
        return d * self.shap_step


class LCODE_HD(LCODE_WD):
    def _calc_distance(self, p: np.array, q: np.array) -> np.array:
        # Hellinger distance
        return np.sqrt(((np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=-1)) / np.sqrt(2)


class LCODE_KS(LCODE_WD):
    def _calc_distance(self, p: np.array, q: np.array) -> np.array:
        # Kolmogorov-Smirnov statistic
        return (np.abs(p.cumsum(axis=-1) - q.cumsum(axis=-1))).max(axis=-1)


if __name__ == '__main__':
    from detectors.base import detection_test
    print('LCODE_WD')
    detection_test(LCODE_WD)

    print('LCODE_HD')
    detection_test(LCODE_HD)

    print('LCODE_KS')
    detection_test(LCODE_KS)
