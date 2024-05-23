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
data = pd.DataFrame(columns=['clients_fraction', 'method', 'n_clients'])

for file in paths:
    with open(file, 'rb') as f:
        res = pickle.load(f)
    df = pd.DataFrame()
    df['clients_fraction'] = res[:-1]
    df['n_clients'] = file.split('_')[-1].split('.')[0]
    df['method'] = file.split('_')[3].split('.')[0]
    data = pd.concat([data, df], ignore_index=True)

data['clients_fraction'] = data['clients_fraction'].astype(np.float32)
data['n_clients'] = data['n_clients'].astype(int)
data = data[data['clients_fraction'] <= 1.0]
# data['clients_fraction'] *= data['n_clients']
sns.pointplot(data=data, x='n_clients', y='clients_fraction', hue='method',
              errorbar=('ci', 95), markers=['^', 'o', 'd', 'p'], capsize=0.1,
              dodge=0.3, linestyle="none", palette='deep', hue_order=['optimal', 'aoi', 'random', 'round'])
plt.legend()
plt.grid()
tikzplotlib.save("fraction.tex")
plt.show()
