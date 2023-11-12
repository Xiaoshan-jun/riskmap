#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:56:25 2023
create hueristic value label for risk map
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
safec = 0.9
dim = 16
num_elements = dim * dim



for i in tqdm(range(0, 10)):
    #create random risky map
    safeplace = int(0.3 * num_elements)
    lowrisk = int(0.4 * num_elements)
    highrisk = num_elements - lowrisk- safeplace
    values_safeplace = np.random.uniform(0, 0, safeplace)
    values_lowrisk = np.random.uniform(0, 0.02, lowrisk)
    values_highrisk = np.random.uniform(1, 1, highrisk)
    combined_values = np.concatenate((values_safeplace, values_lowrisk, values_highrisk))
    np.random.shuffle(combined_values)     # Shuffle the combined values
    UAVmap = combined_values.reshape(dim, dim)
    #print(UAVmap)
    mapname = 'dataset/16risk/map/' + str(dim) + '_' + str(i) + '.npy'
    np.save(mapname, UAVmap)
    grids = []
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
                Hvalue[xI[0]][xI[1]] = manhattanHeuristic(xI, xG)
        for xI in grids:
            xI = (xI[0], xI[1], 1)
            actionList, path, nodeList, count, explored = aStarSearch(xI,xG, UAVmap, safec)
            if path:
                if len(path) > 10:
                    Hvalue2 = Hvalue.copy()
                    for _, node in enumerate(path):
                        Hvalue2[node[0], node[1]] = 0
                    hname = 'dataset/16risk/Hvalue/' + str(dim) + '_' + str(i) + '_' + str(xI) + '_' + str(xG) + '.npy'
                    np.save(hname, Hvalue2)
            else:
                continue
