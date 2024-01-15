from torch.utils.data import Dataset
import os
import numpy as np
import ast
import time
import torch
from tqdm import tqdm
"""
train data includes risk map(graph), destination, and correspond heuristic value map
"""
class CustomDataset(Dataset):
    def __init__(self, graph, targets, i):
        self.graphfilename = graph
        self.labelsfilename = targets
        self.index = i

    def __len__(self):
        return len(self.graphfilename)

    def __getitem__(self, idx):
        return self.graphfilename[idx], self.labelsfilename[idx], self.index[idx]

def dataset_builder(size, split):
    t0 = time.time()
    print("start loading dataset")
    mapdirectory = split + 'map/'
    Hdirectory = split + 'Hvalue/'
    mapfilenames = [f for f in os.listdir(mapdirectory) if f.startswith(str(size))]
    graph = []
    targets = []
    index = []
    count = 1
    for mapfilename in tqdm(mapfilenames):
        # count += 1
        # if count > 2:
        #     break
        print(mapfilename)
        heufilenames = [f for f in os.listdir(Hdirectory) if f.startswith(mapfilename[:-4]+'_')]
        print(heufilenames)
        if heufilenames and len(heufilenames) == 1:
            heufilename = heufilenames[0]
        datapoints = np.load(Hdirectory + heufilename)
        for i in range(int(len(datapoints)/3)):
            #load risk map's name
            #rm = np.load(mapdirectory + mapfilename)
            #rm = np.array(rm).reshape(-1)
            graph.append(mapdirectory + mapfilename)
            #load h value map's name
            #hm = datapoints[f'data_{i}_Hvalue']
            #hm = np.array(hm).reshape(-1)
            targets.append(Hdirectory + heufilename)
            #load start
            #sta = datapoints[f'data_{i}_start']
            #sta = sta[0] * size + sta[1]
            #start.append(sta)
            #load destination
            #des = datapoints[f'data_{i}_destination']
            #des = des[0] * size + des[1]
            #destination.append(des)
            #load index
            index.append(i)

    print("data loading finished, ", "load time: ", time.time() - t0, "number of data: ", len(graph))
    #graph = torch.tensor(graph, dtype=torch.float32)
    #start = torch.tensor(start, dtype=torch.int)
    #destination = torch.tensor(destination, dtype=torch.int)
    #targets = torch.tensor(targets, dtype=torch.float32)
    dataset = CustomDataset(graph, targets, index)
    return dataset