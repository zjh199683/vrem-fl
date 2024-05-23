import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import tikzplotlib

sns.set_palette(sns.color_palette('deep'))
paths = glob('./*.pkl')
paths.sort()
data = pd.DataFrame(columns=['value', 'method', 'metric'])

for file in paths:
    with open(file, 'rb') as f:
        res = pickle.load(f)
    metric = 'bandwidth'
    if 'frac' in file:
        res.pop(0)
        metric = 'clients_fraction'
    df = pd.DataFrame()
    df['value'] = res
    df['metric'] = metric
    df['method'] = file.split('_')[3].split('.')[0]
    data = pd.concat([data, df], ignore_index=True)

data['value'] = data['value'].astype(np.float32)
data = data[data['value'] > 0]
# data['clients_fraction'] = data['clients_fraction'].astype(np.float32)
# data['n_clients'] = data['n_clients'].astype(int)
# data = data[data['clients_fraction'] <= 1.0]
# data['clients_fraction'] *= data['n_clients']
#sns.pointplot(data=data, x='metric', y='value', hue='method', estimator=np.mean,
#              errorbar=('ci', 95), markers=['^', 'o', 'd', 's'], capsize=0.1,
#              dodge=0.3, linestyle="none", palette='deep', hue_order=['optimal', 'aoi', 'random', 'channel'])
sns.stripplot(data=data, x='metric', y='value', hue='method', palette='deep',
            hue_order=['optimal', 'aoi', 'random', 'channel'], dodge=True)
# ax2 = plt.twinx()
# sns.pointplot(data=data[data['metric'] == 'clients_fraction'], x='metric', y='value', hue='method', estimator=np.mean,
#               errorbar=('ci', 95), markers=['^', 'o', 'd'], capsize=0.1,
#               dodge=0.3, linestyle="none", palette='deep', hue_order=['optimal', 'aoi', 'random'], ax=ax2)
plt.legend()
plt.grid()
tikzplotlib.save("bandwidth-taxi.tex")
plt.show()
