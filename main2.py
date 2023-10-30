# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:38:57 2023

@author: dekom
"""
from include.astarastar import aStarSearch
import time
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
import random

random.seed(2)
for a in range(7, 12):
    b = a
    rfactor = 0.033
    x = round(a*0.25)
    y = round(a*0.25)
    width = round(a*0.5)
    width2 = round(b*0.35)
    length = round(a*0.5)
    length2 = round(b/4)
    risk = np.random.rand(a, b)*rfactor/a*5
    for i in range(x, x + width):
        for j in range(y , y + length):
            if i < len(risk) and j < len(risk[0]):
                risk[i][j] = rfactor
      
    for i in range(a):
        risk[0][i] = 0
        risk[a-1][i] = 0
        risk[i][b-1] = 0
    risk = risk.round(2)
    trainfile = 'train' + str(a) + '.txt'
    valfile = 'val' + str(a) + '.txt'
    testfile = 'test' + str(a) + '.txt'
    
    with open(trainfile, 'w') as f, open(testfile, 'w') as f2:
        for i in range(10):
            x = random.randint(0, a-1)
            y = random.randint(0, b-1)
            xG = (x, y)
            x = random.randint(0, a-1)
            y = random.randint(0, b-1)
            xI = (x, y, 1)
            while abs(xG[0] - xI[0]) + abs(xG[1] - xI[1]) < a*0.3:
                x = random.randint(0, a-1)
                y = random.randint(0, b-1)
                xG = (x, y)
            safec = 0.9
            t0 = time.time()
            actionList, path, nodeList, count, explored = aStarSearch(xI,xG, risk, safec)
            if path:
                print(time.time() - t0)
                print(path)
                f.write('\n')
                f.write('risk:')
                np.savetxt(f, risk, fmt='%.2f')
                f.write('start:')
                f.write(str(xI))
                f.write('destination:')
                f.write(str(xG))
                f.write('solution:')
                for a in actionList:
                    f.write(str(a) + ',')
                f2.write('\n')
                f2.write('risk:')
                np.savetxt(f2, risk, fmt='%.2f')
                f2.write('start:')
                f2.write(str(xI))
                f2.write('destination:')
                f2.write(str(xG))
                f2.write('solution:')
                plt.figure(figsize = (8, 8), dpi=100)
                plt.axes()
                for m in range(len(risk)):
                    for n in range(len(risk[0])):
                        my_cmap = cm.get_cmap('Greys')
                        min_val = 0
                        max_val = 0.1
                        norm = matplotlib.colors.Normalize(min_val, max_val)
                        color_i = my_cmap(norm(risk[m][n]))
                        square = plt.Rectangle((m, n), 1, 1, fc=color_i,ec="gray")
                        plt.gca().add_patch(square)
                for p in path:
                    my_cmap = cm.get_cmap('Greys')
                    min_val = 0
                    max_val = rfactor
                    norm = matplotlib.colors.Normalize(min_val, max_val)
                    color_i = my_cmap(norm(risk[m][n]))
                    square = plt.Rectangle((p[0], p[1]), 1, 1, fc=color_i,ec="red", lw = 3)
                    plt.gca().add_patch(square)
                plt.axis('scaled')
                plt.title('searched path from ' + str(xI) + ' to ' + str(xG), fontsize = 20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
            else:
                print(time.time() - t0)
                print(path)
                f.write('\n')
                f.write('risk:')
                np.savetxt(f, risk, fmt='%.2f')
                f.write('start:')
                f.write(str(xI))
                f.write('destination:')
                f.write(str(xG))
                f.write('solution:')
                f.write('[N]')
                f2.write('\n')
                f2.write('risk:')
                np.savetxt(f2, risk, fmt='%.2f')
                f2.write('start:')
                f2.write(str(xI))
                f2.write('destination:')
                f2.write(str(xG))
                f2.write('solution:')
                plt.figure(figsize = (8, 8), dpi=100)
                plt.axes()
                for m in range(len(risk)):
                    for n in range(len(risk[0])):
                        my_cmap = cm.get_cmap('Greys')
                        min_val = 0
                        max_val = 0.1
                        norm = matplotlib.colors.Normalize(min_val, max_val)
                        color_i = my_cmap(norm(risk[m][n]))
                        square = plt.Rectangle((m, n), 1, 1, fc=color_i,ec="gray")
                        plt.gca().add_patch(square)
                square = plt.Rectangle(xI, 1, 1, fc=color_i,ec="red")
                plt.gca().add_patch(square)
                square = plt.Rectangle(xG, 1, 1, fc=color_i,ec="red")
                plt.gca().add_patch(square)
                plt.axis('scaled')
                plt.title('searched path from ' + str(xI) + ' to ' + str(xG), fontsize = 20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

#create random risky map
for i in range(1):
    dim = 64
    num_elements = dim * dim
    num_90_percent = int(0.9 * num_elements)
    num_10_percent = num_elements - num_90_percent
    values_90_percent = np.random.uniform(0, 0.01, num_90_percent)
    values_10_percent = np.random.uniform(0.1, 1, num_10_percent)
    combined_values = np.concatenate((values_90_percent, values_10_percent))
    UAVmap = combined_values.reshape(dim, dim)
    np.random.shuffle(combined_values)     # Shuffle the combined values
    mapname = 'map/' + str(dim) + '_' + str(i) + '.npy'
    np.save(mapname, UAVmap)
    grids = []
    for x in range(dim):
        for y in range(dim):
            grids.append((x, y))
    reservedMap = UAVmap
    for xG in grids:
        Hvalue = np.zeros((dim, dim))
        for xI in grids:
            actionList, path, nodeList, count, explored, visited = aStarSearch(xI,xG, reservedMap)
            if path:
                Hvalue[xI[0]][xI[1]] = len(path) 
            else:
                Hvalue[xI[0]][xI[1]] = -1
        hname = 'Hvalue/' + str(dim) + '_' + str(i) + '_' + str(xG) +'.npy'
        print(Hvalue)
        np.save(hname, Hvalue)