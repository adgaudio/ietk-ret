"""
an analysis of the segmentation model outputs
"""
from matplotlib import pyplot as plt
import os

from model_configs import shared_plotting as SP

base_dir = './data/plots/joint_analysis'
os.makedirs(base_dir, exist_ok=True)

# load the performance data
sep_qualdr = SP.get_separability_scores(('MA', 'HE', 'EX', 'SE', 'OD'))
qualdr = SP.get_qualdr_test_df(True)
sep_idrid = SP.get_separability_scores(('MA', 'HE', 'EX', 'SE', 'OD'))
idrid = SP.get_idrid_test_df(True)
sep_rite = SP.get_separability_scores(('MA', 'HE', 'EX', 'SE', 'OD'))
rite = SP.get_rite_test_df(True)

# Top N tables
N = 1
def save_table(df, fp, new_metric_col_name):
    table = SP.report_topn_models(df, N, new_metric_col_name)
    print(table.to_string())
    print(fp)
    table.to_latex(
        fp, index=True, multirow=True, column_format='|lll',
        bold_rows=True, sparsify=True)


save_table(qualdr, f'{base_dir}/qualdr_top_models.tex', 'MCC')
save_table(idrid, f'{base_dir}/idrid_top_models.tex', 'Dice')
save_table(rite, f'{base_dir}/rite_top_models.tex', 'Dice')

save_table(idrid.loc[['A', 'B', 'D', 'X', 'identity']], f'{base_dir}/idrid_competing.tex', 'Dice')

# Correlation between dice and mcc on test set.
#  left: idrid, right: rite.  show scatter plot with correlation coefficient.
#  'paper/figures/mcc_vs_dice.png'


# Correlations between separability and respective task score.
def corr_catplot(df, xlabel='MCC (QualDR Test)'):
    # scatter plot for first N-1 columns against the Nth column
    Ncol = len(df.columns)-1
    fig, axs = plt.subplots(1, Ncol, figsize=(6, 3), sharex=True, sharey=True)
    last_col = df.columns[-1]
    for ax, col in zip(axs, df.columns):
        SP.correlation_plot(col, last_col, df, ax=ax)
    [ax.set_ylabel('') for ax in axs.ravel()[1:]]
    [ax.set_xlabel('') for ax in axs.ravel()]
    axs[int(Ncol//2)].set_xlabel(xlabel)
    return fig
for xlabel, tdf, sep in [('MCC (QualDR Test)', qualdr, sep_qualdr),
            ('Dice (IDRiD Test)', idrid, sep_idrid),
            ('Dice (RITE Test)', rite, sep_rite)]:
    corr_catplot(tdf.join(sep, how='outer'), xlabel)\
            .savefig(f'{base_dir}/correlation_{xlabel.lower().replace(" ","_").replace("(","").replace(")","")}.png', bbox_inches='tight')



#  fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
#  for ax, col in zip(axs.ravel(), qualdr.join(idrid, how='outer').join(rite, how='outer').join(sep, how='outer')):
    #  SP.correlation_plot(col, 'Averaged Separability (IDRiD)', col, sep, ax)
#  axs[0].set_ylabel('(QualDR Test)')
#  axs[1].set_xlabel('Dice (QualDR Train)')
#  axs[1].set_xlabel('Avg. Lesion Separability Score (IDRiD Train)')

#  Idea is to try to find weighting of lesions that maximizes correlation.  use that weighting in the
#  deep segmentation network loss to re-train best model and show further improvement.
