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
dim = 16
num_elements = dim * dim
for i in tqdm(range(0, 500)):
    #create random risky map
    safeplace = int(0.4 * num_elements)
    lowrisk = int(0.4 * num_elements)
    highrisk = num_elements - lowrisk- safeplace
    values_safeplace = np.random.uniform(0, 0, safeplace)
    values_lowrisk = np.random.uniform(0, 0.05, lowrisk)
    values_highrisk = np.random.uniform(1, 1, highrisk)
    combined_values = np.concatenate((values_safeplace, values_lowrisk, values_highrisk))
    np.random.shuffle(combined_values)     # Shuffle the combined values
    UAVmap = combined_values.reshape(dim, dim)
    print(UAVmap)
    mapname = '16riskMap/' + str(dim) + '_' + str(i) + '.npy'
    np.save(mapname, UAVmap)
    grids = []
    for x in range(dim):
        for y in range(dim):
            grids.append((x, y))
    for xG in tqdm(grids):
        if UAVmap[xG[0]][xG[1]] > 0.1:
            #print('in')
            continue
        Hvalue = np.zeros((dim, dim, 5))
        for xI in grids:
            if UAVmap[xI[0]][xI[1]] > 0.1:
                Hvalue[xI[0]][xI[1]][:] = -1
                continue
            for initialsafety in range(5):
                safetylist = [1, 0.98, 0.96, 0.94, 0.92]
                xI = (xI[0], xI[1], safetylist[initialsafety])
                actionList, path, nodeList, count, explored = aStarSearch(xI,xG, UAVmap, safec)
                if path:
                    Hvalue[xI[0]][xI[1]][initialsafety] = len(path)
                else:
                    Hvalue[xI[0]][xI[1]][initialsafety:] = -1
                    break
        Hvalue[xG[0]][xG[1]] = 0
        hname = '16riskHvalue/' + str(dim) + '_' + str(i) + '_' + str(xG) +'.npy'
        print(Hvalue)
        np.save(hname, Hvalue)