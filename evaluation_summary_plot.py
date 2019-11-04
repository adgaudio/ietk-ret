import glob
import pandas as pd
import seaborn as sns
import os.path


results_dir='./data/evaluation_idrid'

# the bar plot with error bars comparing methods a result across the dataset.
dff = pd.concat([pd.read_csv(fp) for fp in glob.glob(
    os.path.join(results_dir, 'stats_IDRiD_*.csv'))], ignore_index=True)
fig = sns.catplot(
    x='Lesion', y='Statistic', hue='Method', col='Evaluation Method',
    data=dff, kind='bar')
fig.savefig(os.path.join(results_dir, 'evaluation_barplot.png'))
