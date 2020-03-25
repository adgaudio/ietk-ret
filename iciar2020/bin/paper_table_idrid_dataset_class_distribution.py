import numpy as np
import pandas as pd

from simplepytorch.datasets import IDRiD_Segmentation


# get distribution of positive pixels per category, on training set.
c = {}
cols = ('MA', 'HE', 'EX', 'SE', 'OD')
N = 0
def inc(x):
    global N
    N += np.prod(x.shape[:2])
for train_or_test in ['Train', 'Test']:
    dset = IDRiD_Segmentation(
        getitem_transform=IDRiD_Segmentation.as_tensor(
            cols,
            include_image=False, return_numpy_array=True),
        use_train_set=(train_or_test == 'Train')
    )
    counts = np.vstack([inc(x) or x.sum((0,1)) for x in dset])
    c[train_or_test.capitalize()] = counts.sum(0)

# pct diseased pixels 
df = pd.DataFrame(c)
df.index = cols
df.index.name = 'Category'
pct = (df / df.sum()).round(3)
c1 = 'Pos/sum(Pos)'
pct.columns = pd.MultiIndex.from_tuples([(c1, x) for x in pct.columns])
# ratio of pos to negative
pos_neg = ((df/N)).round(4)
c2 = 'Pos/(Pos+Neg)'
pos_neg.columns = pd.MultiIndex.from_tuples([(c2, x) for x in pos_neg.columns])
# count of imgs
df = pct.join(pos_neg)
#  df = pct.join(pd.DataFrame({('Num. Images', 'Train'): [54, 53, 54, 26, 54],
              #  ('Num. Images', 'Test'): [27,27,27,14,27]}, index=cols))
print(df.to_string())
print(df.reset_index().to_latex(index=False, multicolumn=True, sparsify=True)
      .replace(c1, r'\textbf{Pos/$\sum$Pos}')
      .replace(c2, r'\textbf{Pos/(Pos+Neg)}')
      )
print('positive_pixels_per_category', counts.sum(0))
import IPython ; IPython.embed() ; import sys ; sys.exit()

