import warnings
from collections.abc import Callable
import pandas as pd
import numpy as np
import shap


def Wasserstein(p: np.array, q: np.array, **kwargs) -> np.array:
    try:
        step = kwargs['step']
    except KeyError:
        raise KeyError('"step" has not been passed.')
    # 1-Wasserstein distance
    d = np.abs(p.cumsum(axis=-1) - q.cumsum(axis=-1)).mean(axis=-1)
    return d * step


class LCODE(object):
    def __init__(self,
                 metric: Callable = Wasserstein,
                 alpha: float = 2,
                 gamma: float = 0.002,
                 categorical_feat: list = [],
                 include: float or list = 0.99,
                 exclude: list = []):
        """
        Initialize LCODE Class.
        Args:
            metric: Callable function to measure the distance between two distributions.
            alpha: Parameter to make the threshold value of drift detection.
            gamma: Parameter of the smoothing factor of shap distributions.
            categorical_feat: List of index of categorical features.
            include: If this value is float, this value is used as a threshold to determin
                relevant feature automatically. If this value is a list, drift detection runs based on only the features
                included in the list.
            exclude: List of index of features that should be excluded from drift detection.
        """
        self.metric = metric

        # LCODE parameters
        self.alpha = alpha
        self.gamma = gamma
        self.categorical_feat = categorical_feat
        self.include = include
        self.exclude = exclude
        self.n_bin_max = 50
        self.th_min = 0.01 * 0.01

        # variables for drift detection
        self.n_bin = None
        self.explainer = None
        self.tgt_feat_list = None
        self.n_tgt_feat = None
        self.class_list = None
        self.n_class = None
        self.feature_bin_edges = None
        self.shap_bin_edges = None
        self.shap_step = None
        self.shap_bias = None
        self.count = None
        self.d_mean = None
        self.d_var = None
        self.th = None
        self.ref_dist = None
        self.obs_shap_pdf = None
        self.exp_shap_pdf = None

        warnings.filterwarnings("ignore", message="Setting feature_perturbation = \"tree_path_dependent\" because no "
                                "background data was given.")

    def reset(self, classifier, X, y):
        self.explainer = shap.TreeExplainer(classifier)
        shap_values, self.shap_bias = self._get_shap(X)
        self.tgt_feat_list = self._select_relevant_features(shap_values)
        self.n_tgt_feat = len(self.tgt_feat_list)
        self.feature_bin_edges = self._make_feature_bin_edges(X)
        self.shap_bin_edges, self.shap_step = self._make_shap_bin_edges(shap_values)

        self.class_list = np.array(sorted(set(y))).astype(int).tolist()
        self.n_class = len(self.class_list)

        self.count = 0
        self.min_count = np.sqrt(len(X))
        self.d_mean = np.zeros((self.n_tgt_feat, self.n_class))
        self.d_var = np.zeros((self.n_tgt_feat, self.n_class))
        self.th = np.infty

        self.ref_dist = self._make_joint_histograms(X, shap_values)
        self.obs_shap_pdf = np.zeros((self.n_tgt_feat, self.n_class, self.n_bin_max))
        self.exp_shap_pdf = np.zeros((self.n_tgt_feat, self.n_class, self.n_bin_max))

    def update(self, x):
        self.count += 1
        shap_values, _ = self._get_shap(x)
        x_tgt = x[self.tgt_feat_list]
        shap_tgt = shap_values[self.tgt_feat_list]

        self._update_distributions(x_tgt, shap_tgt)
        d = self.metric(self.obs_shap_pdf, self.exp_shap_pdf, step=self.shap_step)
        self._update_threshold(d)
        alert = self._inspect_drift(d)

        return alert, d

    # ==========================================================================
    # Internal methods
    def _update_distributions(self, X, shap_values):
        w = 1 / self.count
        w = max(w, self.gamma)
        f, c = np.indices((self.n_tgt_feat, self.n_class))

        # update observed shapley distributions
        pos = (self.shap_bin_edges < np.expand_dims(shap_values, axis=-1)).sum(axis=-1) - 1
        pos = np.minimum(np.maximum(0, pos), len(self.shap_bin_edges) - 2)
        self.obs_shap_pdf *= (1 - w)
        self.obs_shap_pdf[f.flatten(), c.flatten(), pos.flatten()] += w

        # update expected shapley distributions
        xq = (self.feature_bin_edges < np.expand_dims(X, axis=-1)).sum(axis=-1) - 1
        xq = np.minimum(np.maximum(0, xq), (~np.isnan(self.feature_bin_edges)).sum(axis=1) - 2)
        exp_dist = self.ref_dist[np.arange(len(xq)), :, xq]
        denom = exp_dist.sum(axis=-1)
        denom = np.tile(denom, [exp_dist.shape[-1], 1, 1]).swapaxes(0, 1).swapaxes(1, 2)
        exp_dist = np.divide(exp_dist, denom, out=self.exp_shap_pdf.copy(), where=(denom != 0))
        self.exp_shap_pdf *= (1 - w)
        self.exp_shap_pdf += (w * exp_dist)

    def _inspect_drift(self, d):
        alert = np.array(np.where(self.th < d)).T
        if len(alert):
            alert = [(self.tgt_feat_list[i], self.class_list[j]) for (i, j) in alert]
        else:
            alert = None
        return alert

    def _update_threshold(self, d):
        if self.count > self.min_count:
            # update threshold value only after enough number of samples are passed
            ths = self.d_mean + self.alpha * np.sqrt(self.d_var)
            self.th = ths.max()
            self.th = max(self.th, self.th_min)
        self._update_d_mean_var(d)

    def _update_d_mean_var(self, d):
        n = self.count
        mp = self.d_mean
        vp = self.d_var
        m = ((n - 1) * mp + d) / n
        v = ((n - 1) * (vp + mp**2) + d**2) / n - m**2
        v[v < 0] = 0

        self.d_mean = m
        self.d_var = v

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
        for feature_id in self.tgt_feat_list:
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

        X_ = X[:, self.tgt_feat_list]
        sv = shap_values[:, self.tgt_feat_list]

        xmin = np.nanmin(self.feature_bin_edges, axis=1)
        xmax = np.nanmax(self.feature_bin_edges, axis=1)
        X_ = np.clip(X_, xmin, xmax)
        sv = np.clip(sv, self.shap_bin_edges[0], self.shap_bin_edges[-1])
        for i, _ in enumerate(self.tgt_feat_list):
            jhs = []
            xi = X_[:, i]
            fbe = self.feature_bin_edges[i]
            for c in range(self.n_class):
                jh, _, _ = np.histogram2d(xi, sv[:, i, c], (fbe, self.shap_bin_edges))
                jhs.append(jh)
            jhss.append(np.array(jhs))
        return np.array(jhss)


if __name__ == '__main__':
    pass
