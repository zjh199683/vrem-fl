import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np
import pickle
import glob
import tikzplotlib


sns.set_palette(sns.color_palette('deep'))
# Plot of the estimations
files = glob.glob('./estBitrate_*_optimal_opt_opt_noniid_beta0.pk')
files.append('./optimal_opt_opt_noniid_beta0.pk')
files.sort()
print(files)
data, time, steps, tx_steps = [], [], [], []
for path in files:
    with open(path, 'rb') as f:
        res = pickle.load(f)
        data.append(res['convergence'])
        time.append(res['time'])
        steps.append(res['avgCompSteps'])
        tx_steps.append(res['avgTxSteps'])

plt.figure()
markers = ['*', '.', 'X', 'p', 'd', 'o']
for idx, filename in enumerate(files):
    plt.semilogy(time[idx], data[idx][:len(time[idx])], markers[idx], label=filename)

plt.xlim(580, max([time[i][-1] + 20 for i in range(len(time))]) + 20)
plt.xlabel('Time (s)')
plt.ylabel('Distance from opt')
plt.legend()
plt.grid()
tikzplotlib.save("estimation.tex")
plt.show()

# plot of scheduling methods
files = glob.glob('./estBitrate_250_*_opt_opt_noniid_*beta0.pk')
files.sort()
#files = files[3:]
print(files)
data, time, steps, tx_steps = [], [], [], []
for path in files:
    with open(path, 'rb') as f:
        res = pickle.load(f)
        data.append(res['convergence'])
        time.append(res['time'])
        steps.append(res['avgCompSteps'])
        tx_steps.append(res['avgTxSteps'])

plt.figure()
markers = ['*', '.', 'X', 'p', 'd', 'o']
for idx, filename in enumerate(files):
    plt.semilogy(time[idx], data[idx][:len(time[idx])], markers[idx], label=filename)

plt.xlim(580, max([time[i][-1] + 20 for i in range(len(time))]) + 20)
plt.xlabel('Time (s)')
plt.ylabel('Distance from opt')
plt.legend()
plt.grid()
tikzplotlib.save("scheduling.tex")
plt.show()

# computation methods
files = glob.glob('./estBitrate_250_optimal_*_opt_noniid_beta0.pk')
files.sort()
files = files[::-1]
print(files)
data, time, steps, tx_steps = [], [], [], []
for path in files:
    with open(path, 'rb') as f:
        res = pickle.load(f)
        data.append(res['convergence'])
        time.append(res['time'])
        steps.append(res['avgCompSteps'])
        tx_steps.append(res['avgTxSteps'])

plt.figure()
markers = ['*', '.', 'X', 'p', 'd', 'o']
for idx, filename in enumerate(files):
    plt.semilogy(time[idx], data[idx][:len(time[idx])], markers[idx], label=filename)

plt.xlim(580, max([time[i][-1] + 20 for i in range(len(time))]) + 20)
plt.xlabel('Time (s)')
plt.ylabel('Distance from opt')
plt.legend()
plt.grid()
tikzplotlib.save("computation.tex")
plt.show()

# transmission methods
files = glob.glob('./estBitrate_250_optimal_opt_*_noniid_beta0.pk')
files.sort()
files = files[::-1]
print(files)
data, time, steps, tx_steps = [], [], [], []
for path in files:
    with open(path, 'rb') as f:
        res = pickle.load(f)
        data.append(res['convergence'])
        time.append(res['time'])
        steps.append(res['avgCompSteps'])
        tx_steps.append(res['avgTxSteps'])

plt.figure()
markers = ['*', '.', 'X', 'p', 'd', 'o']
for idx, filename in enumerate(files):
    plt.semilogy(time[idx], data[idx][:len(time[idx])], markers[idx], label=filename)

plt.xlim(580, max([time[i][-1] + 20 for i in range(len(time))]) + 20)
plt.xlabel('Time (s)')
plt.ylabel('Distance from opt')
plt.legend()
plt.grid()
tikzplotlib.save("transmission.tex")
plt.show()
