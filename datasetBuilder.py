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
np.random.seed(80) 
safec = 0.9
dim = 64
num_elements = dim * dim
for i in tqdm(range(0, 500)):
    #create random risky map
    safeplace = int(0.5 * num_elements)
    lowrisk = int(0 * num_elements)
    highrisk = num_elements - lowrisk- safeplace
    values_safeplace = np.random.uniform(0, 0, safeplace)
    values_lowrisk = np.random.uniform(0, 0.05, lowrisk)
    values_highrisk = np.random.uniform(1, 1, highrisk)
    combined_values = np.concatenate((values_safeplace, values_lowrisk, values_highrisk))
    UAVmap = combined_values.reshape(dim, dim)
    print(UAVmap)
    np.random.shuffle(combined_values)     # Shuffle the combined values
    mapname = '64absoluteMap/' + str(dim) + '_' + str(i) + '.npy'
    np.save(mapname, UAVmap)
    grids = []
    for x in range(dim):
        for y in range(dim):
            grids.append((x, y))
    for xG in tqdm(grids):
        Hvalue = np.zeros((dim, dim))
        for xI in grids:
            xI = (xI[0], xI[1], 1)
            actionList, path, nodeList, count, explored = aStarSearch(xI,xG, UAVmap, safec)
            if path:
                Hvalue[xI[0]][xI[1]] = len(path) 
            else:
                Hvalue[xI[0]][xI[1]] = -1
        Hvalue[xG[0]][xG[1]] = 0
        hname = '64absoluteHvalue/' + str(dim) + '_' + str(i) + '_' + str(xG) +'.npy'
        print(Hvalue)
        np.save(hname, Hvalue)