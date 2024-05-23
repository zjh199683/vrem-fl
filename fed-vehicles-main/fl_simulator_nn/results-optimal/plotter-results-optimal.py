import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tikzplotlib
from glob import glob

sns.set_palette(sns.color_palette('deep'))
paths = glob('./*.csv')
paths.sort()
paths = list(np.roll(paths, 1))
data = pd.DataFrame(columns=['round', 'miou', 'n_clients'])

for elem in paths:
    df = pd.DataFrame()
    res = pd.read_csv(elem)
    df['round'] = res['Step']
    df['miou'] = res['Value']
    df['n_clients'] = elem.split('_')[3]
    data = pd.concat([data, df], ignore_index=True)

sns.lineplot(data=data, x='round', y='miou', hue='n_clients')
plt.grid()
plt.legend()
tikzplotlib.save("miou.tex")
plt.show()