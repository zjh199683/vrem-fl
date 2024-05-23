import json
import glob
import numpy as np

filename = 'system_cfg.json'

cfg = dict()
path = '/home/giovanni/Desktop/fed-vehicles/fl_simulator_nn/data/apolloscape/'
roads = ['road01down', 'road02down', 'road03down', 'road04down']

records = [
    [31, 37, 67, 68, 85, 86, 102, 103, 104, 105],
    [2, 3, 4, 5, 6, 18, 19, 20, 21],
    [1, 2, 3, 4, 6, 21, 22, 24, 27, 29, 30, 31, 35],
    [1, 2, 3, 4, 5, 13, 14, 15]
    ]

apollo_records = dict()
for id in range(92):
    idx = 1#np.random.choice(np.arange(0, 4), p=[0.3, 0.2, 0.3, 0.2])
    road = roads[idx]
    record = records[idx]
    apollo_records[id] = [{'road' : road, 'record' : [int(np.random.choice(record))]}]

cfg['mobility'] = True
cfg['min_local_steps'] = 1
cfg['max_local_steps'] = 50
cfg['time_slot'] = 1
cfg['tx_strategy'] = 'opt'
cfg['apollo_records'] = apollo_records

with open(filename, 'w') as f:
    json.dump(cfg, f, indent=4)