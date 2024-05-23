import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tikzplotlib
from glob import glob

sns.set_palette(sns.color_palette('deep'))
paths = glob('./*.csv')
paths.sort()
paths[0], paths[1] = paths[1], paths[0]
data = pd.DataFrame(columns=['round', 'miou', 'method'])

for elem in paths:
    df = pd.DataFrame()
    res = pd.read_csv(elem)
    df['round'] = res['Step']
    df['miou'] = res['Value']
    df['method'] = elem.split('_')[2]
    data = pd.concat([data, df], ignore_index=True)

sns.lineplot(data=data, x='round', y='miou', hue='method')
plt.grid()
plt.legend()
tikzplotlib.save("scheduling-miou.tex")
plt.show()