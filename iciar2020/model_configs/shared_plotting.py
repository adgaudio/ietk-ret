import pandas as pd
import seaborn as sns
import scipy.stats
import numpy as np
import argparse as ap
import re

from simplepytorch.plot_perf import _mode_1_get_perf_data_as_df


def _make_namespace(runid_regex):
    return ap.Namespace(
        runid_regex=runid_regex,
        data_dir='./data',
        index_col='epoch',
        data_fp_regex=r'perf.*\.csv'
    )


def _standardize_method_name(x):
    if '-' in x:  # remove the prefix like Rtest2-
        x = x.split('-', 1)[1]
    return f'avg{x.count("+")+1}:{x}' if '+' in x else x


def get_perf_csvs_as_df(runid_regex):
    ns = _make_namespace(runid_regex)
    print('NNN', ns)
    df = _mode_1_get_perf_data_as_df(ns)
    df.reset_index('filename', drop=True, inplace=True)
    df.rename(index=lambda x: re.sub(r'.*?-', '', x), level='run_id', inplace=True)
    df.index.set_names('Method', level='run_id', inplace=True)
    # --> standardize to naming convention used by separability scores
    df.rename(index=_standardize_method_name, level='Method', inplace=True)
    return df


def get_qualdr_test_df(just_mcc_test_cols=False):
    df = get_perf_csvs_as_df('Qtest2')  # TODO: switch to 3
    if just_mcc_test_cols:
        print(df.columns)
        cols = [x for x in df.columns if x.endswith('_test') and x.startswith('MCC_hard')]
        df = df[cols]
        df.columns = [
            re.sub('MCC_hard_(.*?)_test', r'\1', x)
            for x in df.columns]  # [retinopathy, maculopathy, photocoagulation]
        replace = {'retinopathy': 'DR', 'maculopathy': 'MD', 'photocoagulation': 'PC'}
        df.columns = [replace[x] for x in df.columns]
    return df


def _get_segmentation_test_df(runid_regex, just_dice_test_cols):
    df = get_perf_csvs_as_df(runid_regex)
    if just_dice_test_cols:
        cols = [x for x in df.columns if x.endswith('_test') and x.startswith('Dice_') and x!='Dice_test']
        df = df[cols]
        df.columns = [
            re.sub('Dice_(.*?)_test', r'\1', x)
            for x in df.columns]  # [MA, HE, ...]

    #  df.rename(index=_standardize_method_name, level='Method', inplace=True)
    return df


def get_idrid_test_df(just_dice_test_cols=False):
    return _get_segmentation_test_df('Itest2', just_dice_test_cols)


def get_rite_test_df(just_dice_test_cols=False):
    df = _get_segmentation_test_df('Rtest2', just_dice_test_cols)
    print(df)
    df.columns = [x.capitalize() for x in df]
    return df


def get_separability_scores(lesions=('MA', 'HE', 'EX', 'SE', 'OD')):
    # mean across images, max across lesions and channels
    #  _sep = pd.read_csv('data/histograms_idrid_data/separability_consistency/separability.csv')\
        #  .query('lesion_name in @lesions')\
        #  .groupby(['method_name', 'lesion_name'])[['red', 'green', 'blue']]\
        #  .mean().stack().groupby('method_name').max().rename('Averaged Separability (IDRiD)')
    # mean across lesions and imgs.  max across channels
    _sep = pd.read_csv('data/histograms_idrid_data/separability_consistency/separability.csv')\
        .query('lesion_name in @lesions')\
        .groupby(['method_name'])\
            [['red', 'green', 'blue']].min().stack().groupby('method_name').max().rename('Averaged Separability (IDRiD)')
    _sep.index.rename('Method', inplace=True)
    return _sep


def correlation_plot(col1: str, col2: str, df, ax):
    sns.regplot(col1, col2, df, ax=ax)
    ax.set_title(col1)
    a,b = df[[col2, col1]].dropna().T.values
    a[np.isinf(a)] = 1
    b[np.isinf(b)] = 1
    pearson = scipy.stats.pearsonr(a,b)
    ax.legend([f'$R={np.round(pearson[0], 2):0.2f}$\n$p={np.round(pearson[1], 4):0.4f}$'], loc='lower right')


def report_topn_models(cdfs, N, new_metric_col_name):
    N = 2
    z = cdfs.query('epoch>20').groupby('Method').mean()
    z2 = (z - z.loc['identity'].values)
    top_models = z.stack().groupby(level=-1).nlargest(N).reset_index(level=0, drop=True).rename(new_metric_col_name)
    tm = top_models = top_models.to_frame().join(
        z2.stack().groupby(level=-1).nlargest(N)
        .reset_index(level=0, drop=True).rename('(delta)')
    ).reset_index()
    tm['Task'] = tm.reset_index()['level_1'].values
    table = tm.set_index(['Task', 'Method']).apply(lambda x: '%0.3f (%0.3f)' % (x[new_metric_col_name], x['(delta)']), axis=1).rename(f'{new_metric_col_name} (delta)').reset_index()
    table = table.set_index(['Task', 'Method'])
    return table
