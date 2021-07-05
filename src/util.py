# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import get_whisker_value, load_record, load_record_lcode

MAX_POINTS = 500
LETTER_S = 9
LETTER_M = 10
LETTER_L = 11


def show_drift_map(logfile_path, mode, change_points=[]):
    if mode == 'summary':
        return make_drift_map1(logfile_path, change_points)
    elif mode == 'feature':
        return make_drift_map2(logfile_path, mode, change_points)
    elif mode == 'class':
        return make_drift_map2(logfile_path, mode, change_points)
    else:
        return make_drift_map3(logfile_path)


def make_drift_map1(logfile_path, change_points=[]):
    wd_idx, wd_val, th, drift_map = load_record_lcode(logfile_path)

    idx = list(range(wd_idx[-1]))
    df = pd.DataFrame(index=idx, columns=[''])
    num = np.nanmax(wd_val, axis=(1, 2))
    # denom = th
    # num = np.divide(num, denom, out=np.zeros_like(num).astype(float), where=~np.isnan(denom))

    wd = pd.DataFrame(num, index=wd_idx, columns=[''])
    df.update(wd)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0).T

    skip = (df.shape[1] + (MAX_POINTS - 1)) // MAX_POINTS
    if skip < 1:
        skip = 1
    df = df.iloc[:, ::skip]
    n_labels = int(np.round(min(df.shape[1], MAX_POINTS) // 10, -1))

    _, ax = plt.subplots(1, 1, figsize=(8, 1))
    ax = sns.heatmap(df, xticklabels=n_labels, ax=ax, vmin=0, cbar=False)

    [ax.axvline(i // skip, c='w', ls=':') for i in change_points]

    for _, row in drift_map.iterrows():
        x = row['position'] // skip
        ax.axvline(x, c='gray', zorder=3)

    ax.set_xlabel('Instances')
    ax.set_ylabel('')
    ax.set_yticks([])

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    return df, plt


def make_drift_map2(logfile_path, mode='feature', change_points=[]):
    _, _, _, metadata, _, _, _ = load_record(logfile_path)
    wd_idx, wd_val, th, drift_map = load_record_lcode(logfile_path)
    try:
        data = pd.read_csv('../' + metadata['base_path'])
        idx = data.index
        cols = list(data.columns[:-1]) + ['Th.']
        cols = [c.replace('_', '-') for c in cols]
        shuffled_features = [cols[i] for i in metadata['shuffled_features']]
    except FileNotFoundError:
        idx = list(range(wd_idx[-1]))
        cols = list(range(wd_val.shape[1])) + ['Th.']
        shuffled_features = []

    if mode == 'feature':
        df = pd.DataFrame(index=idx, columns=cols)
        wd = np.nanmax(wd_val, axis=2)
    else:
        cols = list(range(wd_val.shape[2])) + ['Th.']
        df = pd.DataFrame(index=idx, columns=cols)
        wd = np.nanmax(wd_val, axis=1)

    wd = np.concatenate((wd, np.expand_dims(th, axis=1)), axis=1)
    wd = pd.DataFrame(wd, index=wd_idx, columns=cols)
    df.update(wd)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0).T
    
    t, _ = get_whisker_value(wd.iloc[:, :-1])
    vmax = t.max()
    
    skip = (df.shape[1] + (MAX_POINTS - 1)) // MAX_POINTS
    if skip < 1:
        skip = 1
    df = df.iloc[:, ::skip]
    n_labels = int(np.round(min(df.shape[1], MAX_POINTS) // 10, -1))

    _, ax = plt.subplots(1, 1, figsize=(8, 0.8 + len(df) * 0.06))
    ax = sns.heatmap(df, xticklabels=n_labels, ax=ax, vmin=0, vmax=vmax, cbar=False)

    [ax.axhline(i, c='k', lw=1) for i in range(len(df.columns))]
    [ax.axvline(i // skip, c='w', ls=':') for i in change_points]

    for _, row in drift_map.iterrows():
        x = row['position'] // skip
        ax.axvline(x, c='gray', lw=1)
        if mode == 'feature':
            y = row['feature']
        else:
            y = row['class']
        ax.vlines(x, y, y + 1, colors=['cyan'], lw=3, zorder=3)

    ax.set_xlabel('Instances')
    ax.set_ylabel(mode)

    ticklabels = ax.yaxis.get_ticklabels()
    [tl.set_color('r') for tl in ticklabels if tl.get_text() in shuffled_features]
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    return df, plt


def make_drift_map3(logfile_path, n_plot_max=6):
    wd_idx, wd_val, th, drift_map = load_record_lcode(logfile_path)
    indices = wd_idx.tolist()
    drift_map = drift_map.iloc[:n_plot_max]

    if len(drift_map) < 1:
        return drift_map, None

    ncols = 2
    nrows = (len(drift_map) + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(8, 0.8 + nrows), sharex=True, sharey=True)

    axes = axes.flatten()
    for i, (_, row) in enumerate(drift_map.iterrows()):
        pos = row['position']
        f = row['feature']
        c = row['class']
        idx = indices.index(pos)
        v = np.nan_to_num(wd_val[idx].T, 0)

        ax = axes[i]
        sns.heatmap(v, vmin=0, vmax=th[idx], square=True, cbar=False, linecolor='k', linewidths=1, ax=ax)
        ax.scatter(f + 0.5, c + 0.5, marker='o', c='cyan')
        ax.text(0, -0.5, 'Position:%d, Feature:%d, Class:%d' % (pos, f, c))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        if (i // ncols) == (nrows - 1):
            ax.set_xlabel('feature')
        else:
            ax.xaxis.set_visible(False)

        if (i % ncols) == 0:
            ax.set_ylabel('class')
        else:
            ax.yaxis.set_visible(False)

    while(i + 1 < nrows * ncols):
        i += 1
        axes[i].remove()

    return drift_map, plt


def show_class_ratio(dataset_path, mode='area', change_points=None, n_div=200):
    df = pd.read_csv(dataset_path)
    batch_size = int((len(df) / n_div // 50) * 50)
    df.columns = list(df.columns[:-1]) + ['class']
    df['count'] = 1
    df = df.iloc[:, -2:].pivot(columns='class', values='count').fillna(0)
    df['instances'] = (df.index // batch_size) * batch_size
    df = df.groupby(by='instances').mean().T
    n_class = len(df)
    n_instance = df.columns[-1]

    if mode == 'heatmap':
        _, ax = plt.subplots(1, 1, figsize=(8, 0.8 + n_class * 0.1))
        ax = sns.heatmap(df, xticklabels=n_div // 10, ax=ax, vmin=0, vmax=1)
        [ax.axhline(i, c='k', lw=1.5) for i in range(len(df.columns))]
    else:
        _, ax = plt.subplots(1, 1, figsize=(8, 2))
        cmap = sns.color_palette("Blues_r", n_class, as_cmap=True)
        ax = df.T.plot.area(stacked=True, lw=0, ax=ax, cmap=cmap)
        ax.set_xlim(0, int(n_instance * 1.1))
        ax.set_ylim(0, 1)
        ax.legend(prop={'size': LETTER_S})
    ax.set_ylabel('class ratio')

    [ax.axvline((i // batch_size) * batch_size, c='k', ls=':') for i in change_points]

    return df, plt


def set_fontsize(fig, small=12, medium=15, large=18):
    if fig._suptitle:
        fig._suptitle.set_fontsize(large)
    for ax in fig.get_axes():
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(small)
        for item in ([ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(medium)
        for item in ([ax.title]):
            item.set_fontsize(large)


def fix_plot(plt, tight=True, wspace=None, hspace=None, save=False, save_path=None, ext='.png'):
    set_fontsize(plt.gcf(), small=LETTER_S, medium=LETTER_M, large=LETTER_L)
    if tight:
        plt.tight_layout()
    if wspace is not None:
        plt.subplots_adjust(wspace=wspace)
    if hspace is not None:
        plt.subplots_adjust(hspace=hspace)
    if save and save_path:
        plt.gcf().savefig(save_path + ext, dpi=1200)
    plt.show()
