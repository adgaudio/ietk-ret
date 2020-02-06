"""
an analysis of the segmentation model outputs
"""
from matplotlib import pyplot as plt
import os

from model_configs import shared_plotting as SP

base_dir = './data/plots/joint_analysis'
os.makedirs(base_dir, exist_ok=True)


# Correlation between dice and mcc.
#  left: idrid, right: rite.  show scatter plot with correlation coefficient.
#  'paper/figures/mcc_vs_dice.png'


# Correlations between separability and respective task score.
sep = SP.get_separability_scores()
qualdr = SP.get_qualdr_test_df(True)
#  idrid = SP.get_idrid_test_df()
#  rite = SP.get_rite_test_df()


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
tmp = qualdr.join(sep, how='outer')
for ax, col in zip(axs, qualdr.columns):
    SP.correlation_plot(col, 'Averaged Separability (IDRiD)', tmp, ax=ax)
[ax.set_ylabel('') for ax in axs[1:]]
[ax.set_xlabel('') for ax in axs]
axs[1].set_xlabel('MCC (QualDR Test)')
fig.savefig(f'{base_dir}/correlation_sep_mcc_qualdr.png')


#  fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
#  for ax, col in zip(axs.ravel(), qualdr.join(idrid, how='outer').join(rite, how='outer').join(sep, how='outer')):
    #  SP.correlation_plot(col, 'Averaged Separability (IDRiD)', col, sep, ax)
#  axs[0].set_ylabel('(QualDR Test)')
#  axs[1].set_xlabel('Dice (QualDR Train)')
#  axs[1].set_xlabel('Avg. Lesion Separability Score (IDRiD Train)')

#  Idea is to try to find weighting of lesions that maximizes correlation.  use that weighting in the
#  deep segmentation network loss to re-train best model and show further improvement.
