import pickle

from FL_LS_Algorithms import Fed_GD_LS
from utils import *

filename = '../new_corr_datasetEstim_250_2.csv'
print('Importing vehicle data...')
data = import_vehicles_data(filename=filename, fields={'bitrate', 'estimBitrate'}, min_time=600)
IDs = list(data.keys())
data = {k: data[k] for k in IDs}
print('Vehicle data imported')
M = len(data)  # total number of vehicles
time_slot = 1  # duration of one time slot in seconds
reg_lambda = 1e-4  # regularizer
rounds = 30
n = 25  # model params
noniid = True

print('Generating synthetic dataset...')
X, Y = getSyntheticDataset(sizeXUser=100, M=M, n=n, r=n, sigma=1e-5, s_min=-1, s_max=0, noniid=noniid)
print('Dataset generated')

sizePerUser = int(X.shape[1]/M)
X, Y, Ds, Ys = setFL_DS_LSs(X, Y, M)

Ds = {IDs[k]: Ds[k] for k in range(M)}
Ys = {IDs[k]: Ys[k] for k in range(M)}

m = 30  # max number of schedulable clients per round
max_latency = 100 * time_slot  # maximum allowed latency for one round

param_dim = 128  # number of bits
model_size = n * param_dim
scheduling = 'optimal'
comp = 'opt'
tx = 'opt'
aoi = False
save = True

costs, distancesFromOpt, slots, steps, tx_steps = Fed_GD_LS(X, Y, Ds, Ys, model_size, rounds, data, time_slot, m, max_latency,
                                                            mobility=True, comp=comp, scheduling=scheduling, batch_size=1,
                                                            comp_slots_min=1, reg_lambda=reg_lambda, tx=tx, aoi_only=aoi,
                                                            beta=0)
results = dict(time=slots, convergence=distancesFromOpt, avgCompSteps=steps, avgTxSteps=tx_steps)
aoi_str = ''
noniid_str = ''
if noniid:
    noniid_str += '_noniid'
if aoi:
    aoi_str += '_aoi'
if save:
    with open('../estBitrate_250_' + scheduling + '_' + comp + '_' + tx +
              noniid_str + aoi_str + '_beta0' + '.pk', 'wb') as f:
        pickle.dump(results, f)