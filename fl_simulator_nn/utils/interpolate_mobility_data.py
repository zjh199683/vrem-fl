import numpy as np
import pandas as pd
# from datetime import datetime

# Read data
filename = '../../taxi_february.csv' # veh_id,time,position,seconds,x,y
df = pd.read_csv(filename)
data = {}
ids = set(df['veh_id'])
for id in ids:
    sub = df.loc[df['veh_id'] == id]
    time_idx = sub['seconds'].values
    x = sub['x'].values
    y = sub['y'].values

    # # Convert timestamps to datetime.timestamp format
    # time_idx = [datetime.timestamp(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f+01')) for time_str in time_idx]

    # # Round timestamps to second precision
    # time_idx = [round(timestamp) for timestamp in time_idx]

    # Store timestamps and locations
    data[id] = {'time_idx': time_idx, 'x': x, 'y': y}

interp_step = 1  # required time granularity
data_interp = dict.fromkeys(ids)
for id in ids:

    # Interpolate timestamps
    num_timestamps_interp = int((data[id]['time_idx'][-1] - data[id]['time_idx'][0]) // interp_step + 1)
    timestamps_time = [0]
    timestamps_time.extend([data[id]['time_idx'][i+1] - data[id]['time_idx'][0] for i in range(len(data[id]['time_idx'])-1)])
    timestamps_interp = np.interp(range(num_timestamps_interp), timestamps_time, data[id]['time_idx'])

    # Interpolate locations
    x_interp = np.interp(timestamps_interp, data[id]['time_idx'], data[id]['x'])
    y_interp = np.interp(timestamps_interp, data[id]['time_idx'], data[id]['y'])

    # Store interpolated data
    data_interp[id] = {'time_idx': timestamps_interp, 'x': x_interp, 'y': y_interp}  # (key, val) = (veh_id, dict('time_idx', 'x', 'y'))
    # data_interp[id] = [timestamps_interp, x_interp, y_interp]  # (key, val) = (veh_id, (time, x, y)) with time, x and y arrays

data_interp_csv = {'veh_id': [], 'time': [], 'x': [], 'y': []}
for id in ids:
    for timestamp in data_interp[id]['time_idx']:
        data_interp_csv['veh_id'].append(id)
        data_interp_csv['time'].append(timestamp)

    for x in data_interp[id]['x']:
        data_interp_csv['x'].append(x)

    for y in data_interp[id]['y']:
        data_interp_csv['y'].append(y)

df_interp = pd.DataFrame.from_dict(data_interp_csv)
df_interp = df_interp.sort_values('time')
df_interp.to_csv('data_interpolated.csv', index=False)