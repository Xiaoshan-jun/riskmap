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
# risk = []
# risk.append([0.001, 0, 0, 0, 0, 0, 0, 0, 0])
# risk.append([1, 1, 1, 1, 1, 1, 1, 0, 0])
# risk.append([1, 1, 1, 1, 1, 1, 1, 0, 0])
# risk.append([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 1])
# risk.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
# risk.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
# risk.append([0, 0, 0, 0, 0, 0, 1, 1, 0])
# risk.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
# risk.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
random.seed(1)
a = 32
b = 32
rfactor = 0.033
x = round(4)
y = round(0)
width = round(a*0.8)
width2 = round(b*0.35)
length = round(a*0.9)
length2 = round(b/4)
risk = np.random.rand(a, b)*rfactor/a*5
for i in range(x, x + width):
    for j in range(y , y + length):
        if i < len(risk) and j < len(risk[0]):
            risk[i][j] = rfactor
for i in range(x, x + width2):
    for j in range(y , y + length2):
        if i < len(risk) and j < len(risk[0]):
            risk[i][j] = 0
for i in range(x, x + width2):
    for j in range(y + length - length2 , y + length):
        if i < len(risk) and j < len(risk[0]):
            risk[i][j] = 0
for i in range(x + width - width2, x + width):
    for j in range(y , y + length2):
        if i < len(risk) and j < len(risk[0]):
            risk[i][j] = 0
for i in range(x + width - width2, x + width):
    for j in range(y + length - length2 , y + length):
        if i < len(risk) and j < len(risk[0]):
            risk[i][j] = 0   
for i in range(a):
    risk[0][i] = 0
    risk[a-1][i] = 0
    risk[i][b-1] = 0

with open('train.txt', 'w') as f, open('test.txt', 'w') as f2:
    for i in range(1000):
        x = random.randint(0, a-1)
        y = random.randint(0, b-1)
        xI = (x, y, 1)
        x = random.randint(0, a-1)
        y = random.randint(0, b-1)
        xG = (x, y)
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
            np.savetxt(f, risk, fmt='%.3f')
            f.write('start:')
            f.write(str(xI))
            f.write('destination:')
            f.write(str(xG))
            f.write('solution:')
            f.write(str(actionList))
            f.write('\n')
            f2.write('risk:')
            np.savetxt(f2, risk, fmt='%.3f')
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
                    max_val = rfactor
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
            np.savetxt(f, risk, fmt='%.3f')
            f.write('start:')
            f.write(str(xI))
            f.write('destination:')
            f.write(str(xG))
            f.write('solution:')
            f.write('No solution')
            f.write('\n')
            f2.write('risk:')
            np.savetxt(f2, risk, fmt='%.3f')
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
                    max_val = rfactor
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
