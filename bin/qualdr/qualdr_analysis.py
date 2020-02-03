#!/usr/bin/env -S ipython --no-banner -i --
"""
./bin/qualdr_analysis.py --ns 'Qtest2'
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import re
import seaborn as sns

from plot_perf import _mode1_get_frames, bap


def parse_run_id(run_id):
    return re.match(r'Qtest[\d\.]+-(.*)$', run_id).group(1)


def main(ns):
    pass  # TODO


if __name__ == "__main__":
    NS = bap().parse_args()
    print(NS)
    main(NS)

    ns = NS

    dfs = {(run_id, parse_run_id(run_id)): tmpdf
           for (run_id, _), tmpdf in _mode1_get_frames(ns)}
    print(dfs.keys())
    cdfs = pd.concat(dfs, sort=False, names=['run_id', 'method'])

    cdfs.sort_index(level='method', inplace=True)
    cols = [x for x in cdfs.columns if 'MCC_hard' in x]

    # print the best performing models.
    z = cdfs.groupby('method').mean()
    z2 = (z - z.loc['identity'].values)
    # --> plot all models
    ax = z2[cols].plot.bar()
    ax.set_ylabel('MCC (QualDR Test set)')
    ax.set_xlabel('Preprocessing Method')
    ax.figure.tight_layout()
    #  ax.figure.savefig('')  # TODO
    # --> report the top 3 models
    top_models = z[cols].stack().groupby(level=1).nlargest(1).reset_index(level=0, drop=True).rename('MCC')
    tm = top_models = top_models.to_frame().join(
        z2[cols].stack().groupby(level=1).nlargest(1)
        .reset_index(level=0, drop=True).rename('(delta)')
    ).reset_index()
    tm['category'] = tm.reset_index()['level_1'].replace(regex={'MCC_hard_(.*)_test': r'\1'}).values
    table = tm.set_index(['category', 'method']).apply(lambda x: '%0.3f (%0.3f)' % (x['MCC'], x['(delta)']), axis=1)
    print(table.to_string())
    #  table.to_latex('')  # TODO

    # correlate these to the validation scores.
    # TODO

    #  # evaluate the best model confusion matrixes
    #  # --> best runs
    #  run_ids = [(x[0], x[1][0]) for x in cdfs[cols].idxmax().iteritems()]
    #  print(run_ids)
    #  # --> runs with closest perf to identity
    #  #  mrl = method_runid_lookup = dict(cdfs.reset_index('epoch', drop=True).index.swaplevel(0,1).values)
    #  #  assert len(method_runid_lookup) == cdfs.shape[0], "bug: multiple run_ids match same method"
    #  #  run_ids = [(x[0], mrl[x[1]]) for x in z2[cols].drop('identity', axis=0).abs().idxmin().iteritems()]

    #  CMs, CMs_identity = [], []
    #  for col, run_id in run_ids:
    #      CMs.append(pd.HDFStore(f'data/results/{run_id}/perf_matrices_test.h5')[col.replace('MCC_hard', 'CM_soft')])
    #      CMs_identity.append(pd.HDFStore(f'data/results/{run_id.split("-")[0]}-identity/perf_matrices_test.h5')[col.replace('MCC_hard', 'CM_soft')])
    #  a = CMs[0].values
    #  b = CMs_identity[0].values
    #  anorm = a/a.sum(1, keepdims=True)
    #  bnorm = b/b.sum(1, keepdims=True)
    #  assert np.allclose(bnorm[0].sum(), 1)
    #  assert np.allclose(anorm[0].sum(), 1)
    #  plt.figure()
    #  for x in range(a.shape[0]):
    #      plt.plot(anorm[x], label=x)
    #  plt.legend()
    #  plt.figure()
    #  for x in range(a.shape[0]):
    #      plt.plot(bnorm[x], label=x)
    #  plt.legend()

    #  hdf = pd.HDFStore('data/results/Q1-A/perf_matrices_train.h5')

    ##
    ### Correlation to separability score -- cross dataset comparison
    ##
    _sep = pd.read_csv('/home/alex2/s/r/inverted_dehazing/data/histograms_idrid_data/separability_consistency/separability.csv')\
        .query('lesion_name!="OD"')\
        .groupby(['method_name', 'lesion_name'])[['red', 'green', 'blue']]\
        .mean().stack().groupby('method_name').mean().rename('Averaged Separability (IDRiD)')
    _sep.index.rename('method', inplace=True)
    _sep2 = cdfs[cols].groupby('method').mean()

    _sep2.columns = [re.sub('MCC_hard_(.*?)_test', r'\1', x)
                     for x in _sep2.columns]
    _sep2.index = [f'avg{x.count("+")}:{x}' if '+' in x else x for x in _sep2.index]
    sep = _sep2.join(_sep)

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    legend_labels = []
    for ax, col in zip(axs, sep.columns):
        sns.regplot('Averaged Separability (IDRiD)', col, sep, ax=ax)
        ax.set_ylabel('')
        ax.set_title(col.capitalize())
        ax.set_xlabel('')
        a,b = sep[[col, 'Averaged Separability (IDRiD)']].dropna().T.values
        rv = scipy.stats.linregress(b,a)
        ax.legend([f'$R^2$ = {np.round(rv[2], 2)}'])
    axs[0].set_ylabel('MCC (QualDR Test)')
    axs[1].set_xlabel('Avg. Lesion Separability Score (IDRiD Train)')
    #  fig.savefig('')  # TODO
    # conclusions:
    # - suggests better color separation does improve model performance.
    # - but something other color separation is also important for retinopathy grading.
    # - the preprocessing does make the task easier, rather than regularize it.
    # - the pre-processing method is useful on a variety of retinal images
    # - a separability score, which is much faster and simpler to compute than ablative
        # analysis, can be used to guide development of pre-processing methods.
        # We developed the pre-processing methods on the IDRiD train set,
        # guided by the separability scores, and then applied these methods to
        # the QualDR grading task.

"""
    idea: take trained identity model.  get output distribution on a minibatch.  then, "fine-tune" it by trying combinations of inputs that improve perf (maybe just brute force) using botorch?
      --> this approach may naturally favor models that are most like identity
      so it may prevent preprocessing methods that are really good but
      different from identity from being used.  This effect, if it exists, is
      useful since we want images to look like identity so physicians can also look at them.
    idea: do sA+sB on the PickledDict datasets by interpreting the + sign and combining.
"""
