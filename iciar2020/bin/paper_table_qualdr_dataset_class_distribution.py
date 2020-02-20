import pandas as pd ;
from simplepytorch.datasets import QualDR

dfs = {}
for ts in ['train', 'test']:
    d = QualDR(default_set=ts)
    grades = [x[1] for x in d]
    df = pd.DataFrame(grades, columns=['DR', 'MD', 'PC'])
    print('use_train_set =', ts)
    print(df.apply(lambda x: x.value_counts()).to_latex())
    dfs[ts and 'Train' or 'Test'] = df
dff = pd.concat(dfs)
counts = dff.unstack(0).apply(lambda x: x.value_counts())
rv = counts.astype('str') + ' (' + (counts / counts.sum(0)).round(3).astype(str) + '%)'

print(rv.to_latex())


