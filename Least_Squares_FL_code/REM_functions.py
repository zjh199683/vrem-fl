import numpy as np
import scipy.io

#REM = np.loadtxt('dataset_no_sh.txt')
#shadowing = np.loadtxt('FromClusterFiles\shadowMap01.txt')
#shadowing = shadowing[0:300, 0:599]
#sinr = np.zeros((301, 600))

sinr = scipy.io.loadmat('sinr.mat')
sinr = sinr['sinr']

def genGridBSs(numberBSsPerDim, interDistance = 500):
    x = np.linspace(0, 500*(numberBSsPerDim-1), numberBSsPerDim)
    y = np.linspace(0, 500*(numberBSsPerDim-1), numberBSsPerDim)
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    coords = list(zip(X, Y))

    return coords

BSs = genGridBSs(4)


def bitrateThisPos(x, BSs, REM, defaultValue = 256):
    #########
    # --x    --- position of the user
    # --BSs  --- list of base stations positions
    #########
    x = np.asarray(x)
    BSs = np.asarray(BSs)
    # compute closest base station position
    distances = np.linalg.norm(x - BSs, axis = 1)
    indexClosest = np.argmin(distances)
    # if distance from it is greater than 300m, just take a low default sinr value
    distance = distances[indexClosest]
    if distance > 300:
        bitRate = defaultValue*1000
    else:
        # translate to the new center
        xBS = np.asarray(BSs[indexClosest])
        xbar = x - xBS
        # compute angle
        alpha = np.rad2deg(np.arctan(xbar[1]/xbar[0]))
        if xbar[0] < 0:
            alpha = alpha + 180
        elif alpha < 0:
            alpha = alpha + 360
        # now alpha is only positive
        # rotate according to the angle
        if alpha <= 60 or alpha >= 300:
            mySinr = REM[round(xbar[0]), round(xbar[1]+ round(599/2)+1)]
        elif alpha > 60 and alpha <= 180:
            # rotate of 120 degrees
            xr = round(xbar[0]*np.cos(np.deg2rad(-120)) - xbar[1]*np.sin(np.deg2rad(-120)))
            yr = round(xbar[0]*np.sin(np.deg2rad(-120)) + xbar[1]*np.cos(np.deg2rad(-120))) + round(599/2) + 1
            mySinr = REM[xr, yr]
        elif alpha > 180 and alpha < 300:
            # rotate of 240 degrees
            xr = round(xbar[0]*np.cos(np.deg2rad(-240)) - xbar[1]*np.sin(np.deg2rad(-240)))
            yr = round(xbar[0]*np.sin(np.deg2rad(-240)) + xbar[1]*np.cos(np.deg2rad(-240))) + round(599/2) + 1
            mySinr = REM[xr, yr]
    bitRate = sinrToBitRate(mySinr)

    return bitRate

def sinrToBitRate(mySinr):
    sinrLevels = np.asarray([-6.75, -4.96, -2.96, -1.01, 0.96, 2.88, 4.92, 6.70, 8.72, 10.51, 12.45, 14.35, 16.07, 17.88, 19.97])
    cqiInds = np.argwhere(mySinr >= sinrLevels)
    if cqiInds.shape[0] > 0:
        cqi = cqiInds[-1]+1
    else:
        cqi = [0]
    MapCQIToMCS = [0, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28]
    mcs = MapCQIToMCS[cqi[0]]
    McsToItbs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    itbs = McsToItbs[mcs]
    # for now, each user uses 10 subcarriers
    ItbsToBitRate =  [256, 344, 424, 568, 696, 872, 1032, 1224, 1384, 1544, 1736, 2024, 2280, 2536, 2856, 3112, 3240, 3624, 4008, 4264, 4584, 4968, 5352, 5736, 5992, 6200, 7480]
    bitRate = ItbsToBitRate[itbs] # per millisecond
    bitRate = bitRate*1000

    return bitRate

#BSs = [[0, 300], [0, -300]]
#x = [-100, 250]
#
#BR = bitrateThisPos(x, BSs, sinr, defaultValue = 256)
#
#print(BR)

