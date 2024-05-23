from typing import Dict, List
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler


def get_cancer_dataset():

    data = pd.read_csv('../cancer_reg.csv', encoding='ISO-8859-1')
    X = data[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'medIncome', 'popEst2015', 'MedianAge',
              'studyPerCap', 'PercentMarried', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'BirthRate']]
    Y = data['TARGET_deathRate']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = ((Y - Y.mean()) / Y.std()).to_numpy()

    return X.T, Y


def getSyntheticDataset(sizeXUser = 500, M = 32, n = 25, r = 25, sigma = 1e-5,
                        s_min = -2, s_max = 0, noniid = False):
    """ Generate a synthetic dataset.

     :param sizeXUser: how many data points per user
     :param M: how many users
     :param n: model size
     :param r: dimensionality of data generator, i.e., dimension of the generating subspace

     :return X:
     :return Y:
     
    """
    np.random.seed(0)
    N = int(M*sizeXUser)
    F = np.random.randn(n, r)
    # noise
    sigma_x = sigma
    sigma_y = sigma
    Q, R = np.linalg.qr(F)
    s = np.logspace(s_min, s_max, r)
    S = np.diag(s)
    Fbar = np.dot(Q, S)

    # generate true param
    z = np.random.randn(r, 1)
    theta_star = np.dot(Fbar, z)
    Z = np.random.randn(r, N)
    grad = 0.6
    Zbar = grad * Z
    
    myUnb = int(n/2)  # for noniid configuration!
    for i in range(n - myUnb+1):
        Zbar[i:i+myUnb, i*int(N/(n - myUnb + 1)):i*int(N/(n - myUnb + 1)) + int(N/(n - myUnb + 1))] = \
            Z[i:i+myUnb, i*int(N/(n - myUnb + 1)):i*int(N/(n - myUnb + 1)) + int(N/(n - myUnb + 1))]

    if noniid:
        X = np.dot(Fbar, Zbar) + sigma_x * np.random.randn(n, N) / np.sqrt(N)
    else:
        X = np.dot(Fbar, Z) + sigma_x * np.random.randn(n, N) / np.sqrt(N)

    Y = np.dot(X.T, theta_star) + sigma_y * np.random.randn(N, 1)
    Y = Y.squeeze()

    return X, Y


def setFL_DS_LSs(X, Y, M):

    Ds = []
    Ys = []
    index = 0
    sizePerUser = int(X.shape[1]/M)

    for k in range(M):
        Ds.append(X[:, index:index + sizePerUser])
        Ys.append(Y[index:index + sizePerUser])
        index = index + sizePerUser
        if k == 0:
            newX = copy.deepcopy(Ds[k])
            newY = copy.deepcopy(Ys[k])
        else:
            newX = np.hstack((newX, Ds[k]))
            newY = np.hstack((newY, Ys[k]))

    return newX, newY, Ds, Ys


def import_vehicles_data(filename, fields, min_time=0) -> Dict[int, List]:
    """

     Loads the data for the simulation

     :parameter filename: path to the pandas dataframe with data
     :parameter fields: fields to be loaded in the tuple (other than first time index). Iterable or single element.
                        Order will be preserved.

     :returns data: dictionary with (key, value) = (veh ID, (time index, *fields))

    """

    df = pd.read_csv(filename)
    df = df.loc[df['time'] >= min_time]
    ids = set(df['veh_ID'])
    data = {}
    single = isinstance(fields, str)
    for id in ids:
        sub = df.loc[df['veh_ID'] == id]
        time_idx = min(sub['time'].values)
        if single:
            args = sub[fields].values
        else:
            args = dict()
            for field in fields:
                try:
                    args[field] = sub[field].values
                except KeyError:
                    pass
        data[id] = [time_idx, args]

    return data


if __name__ == '__main__':

    path = './veh_info.csv'
    fields = ['x', 'y']

    data = import_vehicles_data(path, fields, min_time=600)
    print(data)