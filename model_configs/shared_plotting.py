import pandas as pd
import seaborn as sns
import scipy.stats
import numpy as np
import argparse as ap
import re

from screendr.plot_perf import _mode_1_get_perf_data_as_df


def _make_namespace(runid_regex):
    return ap.Namespace(
        runid_regex=runid_regex,
        data_dir='./data',
        index_col='epoch',
        data_fp_regex=r'perf.*\.csv'
    )


def get_perf_csvs_as_df(runid_regex):
    ns = _make_namespace(runid_regex)
    df = _mode_1_get_perf_data_as_df(ns)
    df.reset_index('filename', drop=True, inplace=True)
    df.rename(index=lambda x: re.sub(r'Q.*?-', '', x), level='run_id', inplace=True)
    df.index.set_names('Method', level='run_id', inplace=True)
    # --> standardize to naming convention used by separability scores
    df.rename(index=lambda x: f'avg{x.count("+")+1}:{x}' if '+' in x else x, level='Method', inplace=True)
    return df


def get_qualdr_test_df(just_mcc_test_cols=False):
    df = get_perf_csvs_as_df('Qtest2')  # TODO: replace with 3
    if just_mcc_test_cols:
        print(df.columns)
        cols = [x for x in df.columns if x.endswith('_test') and x.startswith('MCC_hard')]
        df = df[cols]
        df.columns = [re.sub('MCC_hard_(.*?)_test', r'\1', x) for x in df.columns]
    return df


def get_idrid_test_df():
    return get_perf_csvs_as_df('Itest1')


def get_rite_test_df():
    return get_perf_csvs_as_df('Rtest1')



def get_separability_scores():
    _sep = pd.read_csv('data/histograms_idrid_data/separability_consistency/separability.csv')\
        .query('lesion_name!="OD"')\
        .groupby(['method_name', 'lesion_name'])[['red', 'green', 'blue']]\
        .mean().stack().groupby('method_name').mean().rename('Averaged Separability (IDRiD)')
    _sep.index.rename('Method', inplace=True)
    return _sep


def correlation_plot(col1: str, col2: str, df, ax):
    sns.regplot(col1, col2, df, ax=ax)
    ax.set_title(col1.capitalize())
    a,b = df[[col2, col1]].dropna().T.values
    pearson = scipy.stats.pearsonr(a,b)
    ax.legend([f'$R={np.round(pearson[0], 2)}$\n$p={np.round(pearson[1], 3)}$'], loc='lower right')
