#!/usr/bin/env python
"""
On IDRiD dataset,
evaluate separability scores for each of the pre-processing methods that we've
generated a histogram for.  Assumes histograms are saved to disk.
Run this after gen_histograms.py.
"""
import numpy as np
import pandas as pd
import glob
import re
import seaborn as sns
from os.path import join, dirname
import os
from matplotlib import pyplot as plt
import seaborn as sns

from ietk.data import IDRiD
from eval_consistency import PATTERN, get_data


def ks_scores_from_hist(a, b, /, is_3d):
    if is_3d:
        raise NotImplemented()
        # this is invalid because depends on order in which axes are raveled
        a = a.ravel()
        b = b.ravel()
        ca = (a/a.sum()).cumsum()
        cb = (b/b.sum()).cumsum()
        return np.abs(ca - cb).max()
    else:
        assert a.shape == (3, 258)
        assert b.shape == (3, 258)
        # --> remove the first and last bin, as those represent data that would
        # be clipped and can skew the KS result.
        a[:, 1] = a[:, 0] ; a[:, -2] = a[:, -1]
        b[:, 1] = b[:, 0] ; b[:, -2] = b[:, -1]
        a = a[:, 1:-1]
        b = b[:, 1:-1]
        assert a.shape == (3, 256)
        assert b.shape == (3, 256)
        ca = (a/a.sum(1, keepdims=True)).cumsum(1)
        cb = (b/b.sum(1, keepdims=True)).cumsum(1)
    return np.abs(ca - cb).max(1, keepdims=True)


def _get_separability_scores(fp, is_3d):
    meta = PATTERN.search(fp).groupdict()
    dat = get_data(fp)
    hists = [dat[x] for x in ['healthy', 'diseased']]
    ks_scores = ks_scores_from_hist(*hists, is_3d=is_3d)
    color = {0: 'red', 1: 'green', 2: 'blue'}
    meta.update({f'{color[ch]}': score for ch, score in enumerate(ks_scores.ravel())})
    return meta


def get_separability_scores(fps, is_3d: bool):
    return pd.DataFrame([_get_separability_scores(fp, is_3d) for fp in fps])


if __name__ == "__main__":
    do_plots = False
    #  for is_3d in ['hist3d-', '']:  # KS test for hist3d doesn't make sense.
    for is_3d in ['']:
        save_img_dir='./data/histograms_idrid_plots/separability_consistency'
        os.makedirs(save_img_dir, exist_ok=True)
        save_csv_fp=f'./data/histograms_idrid_data/separability_consistency/{is_3d}separability.csv'
        os.makedirs(dirname(save_csv_fp), exist_ok=True)
        #  fps = glob.glob(f'./data/histograms_idrid_data/{is_3d}IDRiD*.npz')[:50]  # DEBUG
        fps = glob.glob(f'./data/histograms_idrid_data/{is_3d}IDRiD*.npz')
        fps = [x for x in fps if not any(x.endswith(f'-{y}.npz') for y in ('A2', 'sA+sB2', 'C2', 'C3'))]

        df = get_separability_scores(fps, bool(is_3d))
        df['method_name'] = df['method_name'].apply(lambda x: 'avg%s:%s' % (x.count('+')+1, x) if '+' in x else x)
        df.to_csv(save_csv_fp)
        if not do_plots:
            continue

        print('separability plots')
        # KS score, per channel
        if is_3d:
            df2 = df.rename(columns={'lesion_name': 'Lesion', 'method_name': 'Method'})\
                .melt(['img_id', 'Lesion', 'Method'], ['red'], 'Color Channel', 'Separability score')\
                .sort_values(['Lesion', 'Method'])
        else:
            df2 = df.rename(columns={'lesion_name': 'Lesion', 'method_name': 'Method'})\
                .melt(['img_id', 'Lesion', 'Method'], ['red', 'green', 'blue'], 'Color Channel', 'Separability score')\
                .sort_values(['Lesion', 'Method'])
            fig = sns.catplot(
                x='Lesion', y='Separability score', hue='Method', col='Color Channel', data=df2, kind='bar', palette='tab20c',
            ).savefig(join(save_img_dir, f'{is_3d}separability_per_channel.png'))
        # KS score, mean and max across channels
        fig = sns.catplot(
            x='Lesion', y='Separability score', hue='Method',
            data=df2.groupby(['img_id', 'Lesion', 'Method']).mean().reset_index(),
            kind='bar', palette='tab20c',
        ).savefig(join(save_img_dir, f'{is_3d}separability_mean_of_channels.png'))
        fig = sns.catplot(
            x='Lesion', y='Separability score', hue='Method',
            data=df2.loc[df2.groupby(['img_id', 'Lesion', 'Method'])['Separability score'].idxmax()],
            kind='bar', palette='tab20c',
        ).savefig(join(save_img_dir, f'{is_3d}separability_max_of_channels.png'))
        # --> mean for only best models
        for N in [1,3,4,5]:
            topN_methods = set(
                df2.groupby(['Lesion', 'Method']).mean().reset_index().groupby('Lesion')
                .apply(lambda x: x.nlargest(N, 'Separability score')
                    )['Method'].unique())
            topN_methods.add('identity')
            print(topN_methods)
            df2.groupby(['Method']).mean().plot.bar()\
                .figure.savefig(join(save_img_dir, f'{is_3d}top{N}-separability_mean_of_channels_and_lesions.png'))
            fig = sns.catplot(
                x='Lesion', y='Separability score', hue='Method',
                data=df2.query('Method in @topN_methods').groupby(['img_id', 'Lesion', 'Method']).mean().reset_index(),
                kind='bar', palette='tab20c' if N>1 else sns.color_palette(),
            ).savefig(join(save_img_dir, f'{is_3d}top{N}-separability_mean_of_channels.png'))
        # the proper separability score, as defined in paper, excluding OD
        z = df2.query('Lesion != "OD"').groupby(['Method']).mean()
        z.nlargest(15, 'Separability score').append(z.loc['identity'])\
                .reset_index()\
                .to_latex(join(save_img_dir, f'top15-separability.tex'), index=False)
