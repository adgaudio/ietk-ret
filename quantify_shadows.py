"""
This script evaluates whether the amount of shadow in an image correlates
to the DR grade assigned by physicians.  Conclusions at bottom.
"""
import dehaze
from joblib import Parallel, delayed
import glob
import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns


# get image files to analyze
fps_dct = {
    0: glob.glob('./data/messidor_healthy/*/*'),
    1: glob.glob('./data/messidor_grade1/*/*'),
    2: glob.glob('./data/messidor_grade2/*/*'),
    3: glob.glob('./data/messidor_grade3/*/*'),
}


# For each image, fetch shadow size information from the dark channel of
# illumination step
def job(fp):
    ill, deh = dehaze.illuminate_from_fp(fp)
    # try to quantify amount of shadow
    return {'fp': fp,
            'shadow_size': ill['t_refined'][~ill['background'][:, :, 0]].sum()
            }

fp_csv = 'quantify_shadows.csv'
if not os.path.exists(fp_csv):
    with Parallel(n_jobs=-1) as parallel:
        results = {
            key: parallel(delayed(job)(fp) for fp in fps)
            for key, fps in fps_dct.items()}

    df = pd.DataFrame([dict(grade=k, **vv)
                    for k, v in results.items() for vv in v])
    df.to_csv(fp_csv, index=False)
else:
    df = pd.read_csv(fp_csv)


# Generate plots of this data.

# Bar plot
f, ax = plt.subplots(1, 1, num=1)
df.groupby('grade')['shadow_size'].mean().plot.bar(
    title="Amount of shadow for each DR grade.  grade>0 is diseased", ax=ax)
ax.hlines(df.query('grade==0')['shadow_size'].median(), -1, 4, label='median shadow size, healthy', color='red')
ax.hlines(df.query('grade>0')['shadow_size'].median(), -1, 4, label='median shadow_size, diseased', color='green')
ax.legend()
# if bar plot showed that shadow size decreased as grade increased (ie that
# healthy images have more shadows), I'd be inspired to suspect that healthy
# images hide disease.  But that isn't the case.  In fact, higher grades have
# more shadows!

# However, bar plots can hide details.  Next step is to look at the
# distribution of shadow sizes for each grade.


# To do this, I create violinplot (ie histogram) for each grade to define
# a useful percentile bin width.  It is immediately clear that the middle
# grades have the least shadow.  Maybe what we're seeing is physicians look
# extra carefully at images with shadows, are somewhat uncertain about the
# diagnosis, and give the images a grade somewhere in the middle.
# But in general, these findings don't support my initial hypothesis
pd.cut(df['shadow_size'], 100)
# that healthy images are more shadowed.
plt.figure(2)
sns.violinplot('grade', 'shadow_size', data=df)

# Then for each grade,  group the shadow sizes into 100 bins.
# and plot the bins vs mean bin value, colored by grade.
# Since all grades overlap each other, the results suggest all grades
# probably come from the same distribution
f, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for ax, bins in zip(axs.ravel(), [10, 25, 50, 100]):
    df['shadow_size_bin'] = pd.cut(df['shadow_size'], bins).apply(lambda x: float(x.right)).astype('float')
    df['shadow_size_bin'] = df['shadow_size_bin'] / df['shadow_size_bin'].max()
    df2 = df.groupby(['grade', 'shadow_size_bin'])['shadow_size'].mean().reset_index()
    df2.plot.scatter('shadow_size_bin', 'shadow_size', c='grade', ax=ax, title='%s bins' % bins)
f.suptitle('Quantile bins vs shadow_size.  Each plots shows 4 grades.\n All 4 plots suggest grades have same distribution')
f.tight_layout()


plt.show()

# conclusion 1: the sum of shadow pixel intensities in the illumination
# corrected depth map (`t_refined`) of an image does not appear to have much
# relationship to the grade assigned by physicians.  On one hand, it is
# comforting to see that shadows (at least in the way I defined them) likely
# aren't a problem for physicians.  On the other, I need a new hypothesis!

# conclusion 2:  Healthy images have less shadows than diseased images most of
# the time.  This can be seen by the horizontal lines on the bar plots.
