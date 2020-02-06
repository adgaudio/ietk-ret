#!/usr/bin/env -S ipython --no-banner -i --
"""
./bin/qualdr_analysis.py --ns 'Qtest2'
"""
from matplotlib import pyplot as plt
import os
import re

import model_configs.shared_plotting as SP


def main(ns):
    pass  # TODO


if __name__ == "__main__":
    #  NS = bap().parse_args()
    #  print(NS)
    #  main(NS)
    #  ns = NS

    base_dir = './data/plots/qualdr'
    os.makedirs(base_dir, exist_ok=True)

    cdfs = SP.get_qualdr_test_df()
    print(cdfs.head(2))
    cdfs.sort_index(level='Method', inplace=True)
    cols = [x for x in cdfs.columns if 'MCC_hard' in x]

    # print the best performing models.
    z = cdfs.groupby('Method').mean()
    z2 = (z - z.loc['identity'].values)
    # --> plot all models
    ax = z2[cols].plot.bar()
    #  ax.set_ylabel('MCC (QualDR Test set)')
    ax.set_ylabel('Delta MCC on QualDR Test\n(relative difference from identity)')
    ax.set_xlabel('Preprocessing Method')
    ax.figure.tight_layout()
    ax.figure.savefig(f'{base_dir}/qualdr_mcc_top_models.png')
    # --> report the top 3 models
    N = 3
    top_models = z[cols].stack().groupby(level=1).nlargest(N).reset_index(level=0, drop=True).rename('MCC')
    tm = top_models = top_models.to_frame().join(
        z2[cols].stack().groupby(level=1).nlargest(N)
        .reset_index(level=0, drop=True).rename('(delta)')
    ).reset_index()
    tm['Category'] = tm.reset_index()['level_1'].replace(regex={'MCC_hard_(.*)_test': r'\1'}).values
    table = tm.set_index(['Category', 'Method']).apply(lambda x: '%0.3f (%0.3f)' % (x['MCC'], x['(delta)']), axis=1).rename('MCC (delta)').reset_index()
    table = table.replace({'retinopathy': 'DR', 'maculopathy': 'MD', 'photocoagulation': 'PC'}).set_index(['Category', 'Method'])
    print(table.to_string())
    table.to_latex(f'{base_dir}/qualdr_mcc_top_models.tex', multirow=True, column_format='|lll')

    # correlate these to the validation scores.
    # TODO

    #  # evaluate the best model confusion matrixes
    #  # --> best runs
    #  run_ids = [(x[0], x[1][0]) for x in cdfs[cols].idxmax().iteritems()]
    #  print(run_ids)
    #  # --> runs with closest perf to identity
    #  #  mrl = Method_runid_lookup = dict(cdfs.reset_index('epoch', drop=True).index.swaplevel(0,1).values)
    #  #  assert len(Method_runid_lookup) == cdfs.shape[0], "bug: multiple run_ids match same Method"
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
