from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import cm, colormaps
import matplotlib.pyplot as plt
import math


def hard_vs_rand_negatives(runs, series, draw_optimal=False, legend=True):
    for run, s in zip(runs, series):
        if run.data.params['nce_sampling_strategy'] == 'knn': continue
        #     if int(run.data.params['n_inters']) < 3.5e6: continue

        ax = s.plot(color='b', alpha=1, lw=2, marker='o' if len(s) == 1 else '')

    pop_line = ax.get_lines()[-1]

    plotted_hard = False
    for run, s in zip(runs, series):
        if run.data.params['nce_sampling_strategy'] == 'popularity': continue
        ax = s.plot(color='r', alpha=1, marker='o' if len(s) == 1 else '')
        plotted_hard = True

    if plotted_hard:
        hard_line = ax.get_lines()[0]


    if plotted_hard:
        lines = [pop_line, hard_line]
        labels = ['Random negatives', 'Hard negatives']
    else:
        lines = [pop_line]
        labels = ['Random negatives']

    if draw_optimal:
        optimal = get_optimal(series)
        ax = optimal.plot(lw=1, color='k', logx=True)
        max_line = ax.get_lines()[-1]

        lines.append(max_line)
        labels.append('Optimal performance')

    plt.xscale('log')
    plt.xlabel('Total train FLOPs')
    plt.ylabel('NDCG @ 5')
    plt.title('NDCG as a function of total train FLOPs')
    if legend: plt.legend(lines, labels)
    plt.grid()


class ColorGetter(Enum):
    N_INTERS = 1
    NON_EMB_PARAMS = 2
    TOTAL_PARAMS = 3
    DATA_OVER_PARAMS = 4

    def get(self, run):
        if self == ColorGetter.N_INTERS:
            return np.log10(int(run.data.params['n_inters']))
        elif self == ColorGetter.NON_EMB_PARAMS:
            return np.log10(int(run.data.params['model_non_emb_params']))
        elif self == ColorGetter.TOTAL_PARAMS:
            return np.log10(int(run.data.params['model_total_params']))
        elif self == ColorGetter.DATA_OVER_PARAMS:
            return np.log10(int(run.data.params['n_inters']) / int(run.data.params['model_non_emb_params']))
        else:
            raise ValueError()

    def __str__(self):
        if self == ColorGetter.N_INTERS:
            return 'n_inters'
        elif self == ColorGetter.NON_EMB_PARAMS:
            return 'non_emb_params'
        elif self == ColorGetter.TOTAL_PARAMS:
            return 'model_total_params'
        elif self == ColorGetter.DATA_OVER_PARAMS:
            return r'$log(\frac{dataset\ size}{model\ params})$'
        else:
            raise ValueError()


def colored_scaling(runs, series, color_getter: ColorGetter, full_title=False, draw_optimal=True, highlight_func=None):
    color_vals = sorted(set([color_getter.get(run) for run in runs]))
    max_inters = max(color_vals)
    min_inters = min(color_vals)

    cmap = colormaps['turbo']
    for run, s in zip(runs, series):
        if highlight_func and highlight_func(run): continue
        v = color_getter.get(run)
        color = (v - min_inters) / (max_inters - min_inters)
        s.plot(color=cmap(color), alpha=0.5)

    if highlight_func:
        for run, s in zip(runs, series):
            if not highlight_func(run): continue
            v = color_getter.get(run)
            color = (v - min_inters) / (max_inters - min_inters)
            s.plot(color=cmap(color), marker='o')

    if draw_optimal:
        optimal = get_optimal(series)
        ax = optimal.plot(lw=1, color='k', logx=True)
        max_line = ax.get_lines()[-1]

    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array(color_vals)
    plt.colorbar(sm, ax=plt.gca())

    plt.xlabel('Total train FLOPs')
    plt.ylabel('NDCG @ 5')
    plt.xscale('log')
    if full_title:
        plt.title(f'NDCG as a function of total train FLOPs\nColor represents{color_getter}')
    else:
        plt.title(f'Color represents {color_getter}')


def get_optimal(series):
    df = pd.DataFrame({f's_{i}': s for i, s in enumerate(series)})
    _interpolate_df(df)
    return df.max(1)


def _interpolate_df(df):
    for col in df.columns:
        s = df[col]
        sint = s.interpolate(method='index')
        sint[s[~s.isna()].index[-1]:] = np.nan
        df[col] = sint


def hard_optimal_vs_rand_optimal(runs, series):
    pop_series = [s for i, s in enumerate(series) if
                  runs[i].data.params['nce_sampling_strategy'] == 'popularity']
    rand_best = get_optimal(pop_series)

    hard_series = [s for i, s in enumerate(series) if runs[i].data.params['nce_sampling_strategy'] == 'knn']
    hard_best = get_optimal(series)

    ref = pd.DataFrame({'best hard ndcg': hard_best, 'best rand ndcg': rand_best})
    _interpolate_df(ref)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    ref.plot(logx=True, ax=plt.gca())
    plt.grid()
    plt.subplot(122)
    (ref['best hard ndcg'] / ref['best rand ndcg']).plot(logx=True)


def get_lift_df(runs, series):
    pop_series = [s for i, s in enumerate(series) if
                  runs[i].data.params['nce_sampling_strategy'] == 'popularity']
    rand_best = get_optimal(pop_series)

    docs = []
    extra_params = [
        'nce_knn_discard_neighbors',
        'nce_knn_hard_negatives_pool_size',
        'nce_knn_random_frac',
        'nce_num_negatives',
    ]

    for run, s in zip(runs, series):
        if len(s) == 1: continue
        if run.data.params['nce_sampling_strategy'] == 'popularity': continue
        df = pd.DataFrame({'hard': s, 'ref': rand_best})
        _interpolate_df(df)

        doc = df.dropna().sort_values('hard', ascending=False).iloc[0].to_dict()
        doc['model_total_params'] = int(run.data.params['model_total_params'])
        doc['model_non_emb_params'] = int(run.data.params['model_non_emb_params'])
        doc['n_inters'] = int(run.data.params['n_inters'])
        for col in extra_params:
            doc[col] = float(run.data.params[col])
        docs.append(doc)

    df = pd.DataFrame(docs)
    df['lift'] = df['hard'] / df['ref']
    df['params / data'] = df['model_total_params']  / df['n_inters']
    df['log_n_inters'] = np.log10(df['n_inters'])
    return df
