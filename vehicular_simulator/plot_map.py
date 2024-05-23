import smopy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd

north, east, south, west = 45.4635, 11.7797, 45.3507, 11.9916
def map_to_coordinates(stations):

    y_min = south
    y_max = north
    x_min = east
    x_max = west

    max_x_meters = 111.139 * (x_max - x_min) * 1e3
    max_y_meters = 111.139 * (y_max - y_min) * 1e3

    mask = np.ones(stations.shape[0])
    for i in range(stations.shape[0]):
        if stations[i][0] > max_x_meters + 600 or stations[i][1] > max_y_meters + 600:
            mask[i] = 0
    stations = np.array([stations[i] for i in range(stations.shape[0]) if mask[i]])
    new_stations = np.zeros(stations.shape)
    for i, station in enumerate(stations):
        new_stations[i] = [station[0] * 1e-3 / 111.139 + x_min,
                           station[1] * 1e-3 / 111.139 + y_min]

    return new_stations

np.random.seed(0)
map = smopy.Map((north, east, south, west), z=12)

#stations = np.load('../BSsNumpy.npy')
#stations = map_to_coordinates(stations)

df1 = pd.read_csv(
        "./208.csv",
        names=[
            "Radio",
            "MCC",
            "Net",
            "Area",
            "Cell",
            "unit",
            "lon",
            "lat",
            "range",
            "samples",
            "changeble",
            "created",
            "updated",
            "?",
        ],
    )
df2 = pd.read_csv(
        "./222.csv",
        names=[
            "Radio",
            "MCC",
            "Net",
            "Area",
            "Cell",
            "unit",
            "lon",
            "lat",
            "range",
            "samples",
            "changeble",
            "created",
            "updated",
            "?",
        ],
    )
data = pd.concat([df1[df1['Radio'] == 'LTE'], df2[df2['Radio'] == 'LTE']], ignore_index=True)
del df1
del df2
data = data.loc[(data['lon'] < west) & (data['lon'] > east) & (data['lat'] < north)
                & (data['lat'] > south)]
stations = data[['lon', 'lat']].to_numpy()
ax = map.show_mpl()
points = np.zeros(stations.shape)
for idx, bs in enumerate(stations):
    x, y = map.to_pixels(bs[1], bs[0])
    # ax.plot(x, y, 'or', ms=10, mew=6)
    points[idx] = [x, y]
vor = Voronoi(points)
voronoi_plot_2d(vor, point_size=1.5, line_width=0.5, line_alpha=0.7, ax=ax, show_vertices=False)
plt.show()