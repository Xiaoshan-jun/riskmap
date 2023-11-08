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
safec = 0.8
dim = 64
num_elements = dim * dim
for i in tqdm(range(1, 1080)):
    #create random risky map
    mapname = '64map/' + str(64) + '_' + str(i) + '.npy'
    UAVmap = np.load(mapname)
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
        hname = '64Hvalue/' + str(dim) + '_' + str(i) + '_' + str(xG) +'.npy'
        print(Hvalue)
        np.save(hname, Hvalue)