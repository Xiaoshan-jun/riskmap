#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:56:25 2023
create hueristic value label for wind map
@author: jxiang9143
"""
from include.astarastar import aStarSearch
import time
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
from include.astarastar import manhattanHeuristic

np.random.seed(80) 
safec = 0.8
dim = 16
num_elements = dim * dim
for i in tqdm(range(98, 1080)):
    mapname = 'dataset/16wind/map/' + str(dim) + '_' + str(i) + '.npy'
    UAVmap = np.load(mapname)
    grids = []
    datapoints = []
    for x in range(dim):
        for y in range(dim):
            grids.append((x, y))
    for xG in tqdm(grids):
        if UAVmap[xG[0]][xG[1]] > 0.1:
            #print('in')
            continue
        Hvalue = np.zeros((dim, dim))
        for xI in grids:
            if UAVmap[xI[0]][xI[1]] > 0.1:
                Hvalue[xI[0]][xI[1]] = -1
            else:
                xI2 = (xI[0], xI[1], 1)
                actionList, path, nodeList, count, explored = aStarSearch(xI2,xG, UAVmap, safec)
                if actionList:
                    Hvalue[xI[0]][xI[1]] = manhattanHeuristic(xI, xG)
                else:
                    Hvalue[xI[0]][xI[1]] = -1
        for xI in grids:
            xI = (xI[0], xI[1], 1)
            if manhattanHeuristic(xI, xG) < 10 or Hvalue[xI[0]][xI[1]] == -1:
                continue
            actionList, path, nodeList, count, explored = aStarSearch(xI,xG, UAVmap, safec)
            if path and actionList and nodeList:
                if len(path) > 16:
                    Hvalue2 = Hvalue.copy()
                    for _, node in enumerate(path):
                        Hvalue2[node[0], node[1]] = 0
                    data = {}
                    data['start'] = xI
                    data['destination'] = xG
                    data['Hvalue'] = Hvalue2
                    datapoints.append(data)
            else:
                continue
    kwargs = {}
    for j, datapoint in enumerate(datapoints):
        for key, array in datapoint.items():
            kwargs[f'data_{j}_{key}'] = array
    hname = 'dataset/16wind/Hvalue/' + str(dim) + '_' + str(i) + '_' + str(xI) + '_' + str(xG) + '.npz'
    np.savez(hname, **kwargs)
