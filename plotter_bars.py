import glob
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import tikzplotlib

sns.set_palette(sns.color_palette('deep'))
files = glob.glob('./estBitrate_250_optimal_opt_*_noniid_beta0.pk')

data = pd.DataFrame(columns=['comp', 'type', 'value'])
for path in files:
    with open(path, 'rb') as f:
        res = pickle.load(f)
    comp = 'tx'
    if 'optimal_opt_opt' in path:
        type = 'opt'
    elif 'min_tx' in path:
        type = 'min'
    else:
        type = 'max'
        maxres = np.mean(res['avgTxSteps'])
    for elem in res['avgTxSteps']:
        data = pd.concat([data, pd.DataFrame(dict(comp=comp, type=type, value=elem), index=[0])],
                         ignore_index=True)

data['value'] = data['value'] / maxres
files = glob.glob('./estBitrate_250_optimal_*_opt_noniid_beta0.pk')

for path in files:
    with open(path, 'rb') as f:
        res = pickle.load(f)
    comp = 'cpu'
    if 'optimal_opt_opt' in path:
        type = 'opt'
    elif 'min' in path:
        type = 'min'
    else:
        type = 'max'
        maxres = np.mean(res['avgCompSteps'])
    for elem in res['avgCompSteps']:
        data = pd.concat([data, pd.DataFrame(dict(comp=comp, type=type, value=elem), index=[0])],
                         ignore_index=True)

data['value'][data['comp'] == 'cpu'] /= maxres #= data.loc[data['comp'] == 'cpu']['value'] / maxres
ax = sns.barplot(data=data, x='comp', y='value', hue='type')
for i in ax.containers:
    ax.bar_label(i,)
plt.grid()
tikzplotlib.save('resources.tex')
plt.show()