from torch.utils.data import Dataset
import os
import numpy as np
import ast
import time
import torch
"""
train data includes risk map(graph), destination, and correspond heuristic value map
"""
class CustomDataset(Dataset):
    def __init__(self, graph, destination, targets):
        self.data = graph
        self.des = destination
        self.labels = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.des[idx], self.labels[idx]

def dataloader(size, split):
    t0 = time.time()
    print("start loading dataset")
    if split == 1:
        mapdirectory = 'train/map/'
        Hdirectory = 'train/Hvalue/'
    elif split == 2: #for evaluate
        mapdirectory = 'map/'
        Hdirectory = 'Hvalue/'
    else:
        mapdirectory = 'test/map/'
        Hdirectory = 'test/Hvalue/'
    mapfilenames = [f for f in os.listdir(mapdirectory) if f.startswith(str(size))]
    graph = []
    destination = []
    targets = []
    for mapfilename in mapfilenames:
        heufilenames = [f for f in os.listdir(Hdirectory) if f.startswith(str(mapfilename[:-4] + '_'))]
        for heufilename in heufilenames:
            #load risk map
            rm = np.load(mapdirectory + mapfilename)
            rm = np.array(rm).reshape(-1)
            graph.append(rm)
            #load h value map
            hm = np.load(Hdirectory + heufilename)
            hm = np.array(hm).reshape(-1)
            targets.append(hm)
            #load destination
            prelen = len(mapfilename[:-4])
            des = heufilename[prelen+1: - 4]
            des = ast.literal_eval(des)
            des = des[0] * size + des[1]
            destination.append(des)
    print("data loading finished, ", "load time: ", time.time() - t0, "number of data: ", len(graph))
    graph = torch.tensor(graph, dtype=torch.float32)
    destination = torch.tensor(destination, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.float32)
    dataset = CustomDataset(graph, destination, targets)
    return dataset