import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datasets = ['./newDataset.csv', './bitrate_realmap.csv', './bitrate_realmap_600_std6.csv',
            './bitrate_realmap_600_std8.csv', './new_bitrate_realmap_600_std6.csv',
            './new_bitrate_realmap_600_std6_interf07.csv']
dataframes = [pd.read_csv(elem) for elem in datasets]
df = pd.DataFrame(columns=['map', 'bitrate'])
maps = ['original', '5g', '5g_6', '5g_8', 'newest', 'interference']
for i, frame in enumerate(dataframes):
    df = pd.concat([df, pd.DataFrame({'map' : maps[i], 'bitrate' : frame['bitrate'] / 1e6})])


sns.violinplot(data=df, x='map', y='bitrate')
plt.ylim([0, 9])
plt.show()