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
import random
from tqdm import tqdm
np.random.seed(42) 
safec = 0.9
dim = 16
num_elements = dim * dim
for i in tqdm(range(45, 100)):
    #create random risky map
    safeplace = int(0.4 * num_elements)
    lowrisk = int(0.4 * num_elements)
    highrisk = num_elements - lowrisk- safeplace
    values_safeplace = np.random.uniform(0, 0, safeplace)
    values_lowrisk = np.random.uniform(0, 0.05, lowrisk)
    values_highrisk = np.random.uniform(0.1, 1, highrisk)
    combined_values = np.concatenate((values_safeplace, values_lowrisk, values_highrisk))
    UAVmap = combined_values.reshape(dim, dim)
    np.random.shuffle(combined_values)     # Shuffle the combined values
    mapname = 'map/' + str(dim) + '_' + str(i) + '.npy'
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
        hname = 'Hvalue/' + str(dim) + '_' + str(i) + '_' + str(xG) +'.npy'
        print(Hvalue)
        np.save(hname, Hvalue)