#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:46:00 2023
divideData.py divide data into train and test
@author: jxiang9143
"""
import os
import shutil
from tqdm import tqdm
import random
#--parameters
dataset = "16risk"
size = 16
percentageOfVal = 0.1
#--
mapdirectory = dataset + '/map/'
Hdirectory = dataset + '/Hvalue/'
mapfilenames = [f for f in os.listdir(mapdirectory) if f.startswith(str(size))]
random.shuffle(mapfilenames)
split = int(percentageOfVal*len(mapfilenames))
mapfilenamesVal = mapfilenames[:split]
mapfilenamesTrain = mapfilenames[split:]

for mapfilename in tqdm(mapfilenamesTrain):
    heufilenames = [f for f in os.listdir(Hdirectory) if f.startswith(mapfilename[:-4]+'_')]
    if heufilenames and len(heufilenames) == 1:
        heufilename = heufilenames[0]
    else:
        continue   
    newmapdirectory = dataset + '/train/map/'
    newheudirectory = dataset + '/train/Hvalue/'
    shutil.move(mapdirectory + mapfilename, newmapdirectory)
    shutil.move(Hdirectory + heufilename, newheudirectory)
    
for mapfilename in tqdm(mapfilenamesVal):
    heufilenames = [f for f in os.listdir(Hdirectory) if f.startswith(mapfilename[:-4]+'_')]
    if heufilenames and len(heufilenames) == 1:
        heufilename = heufilenames[0]
    else:
        continue   
    newmapdirectory = dataset + '/val/map/'
    newheudirectory = dataset + '/val/Hvalue/'
    shutil.move(mapdirectory + mapfilename, newmapdirectory)
    shutil.move(Hdirectory + heufilename, newheudirectory)
