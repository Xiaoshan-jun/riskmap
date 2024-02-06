import argparse
from include.dataloader_large_data import dataset_builder
from torch.utils.data import DataLoader
from include.model import Graph2HeuristicModel
from tqdm import tqdm
import torch
from include.astarastar import aStarSearch
import time
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
import time
def manhattanHeuristic(state, goal):
   """ewg, newc)
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.
   """
   return abs(goal[0] - state[0]) + abs(goal[1] - state[1])

##Dataset params
parser = argparse.ArgumentParser(description='Train TrajAirNet model')
parser.add_argument('--map_size', type=int, default= 16) #map is map_size * map_size
parser.add_argument('--block_size', type=int, default= 16*16) #must equal to map_size**2 #block_size is the number of nodes
#model parameter
parser.add_argument('--n_embd', type=int, default= 864)  #embedded item size
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--dropout', type=int, default=0.2)
parser.add_argument('--n_head', type=int, default=6) #number of head, decide how many head the channels devided into.
#train parameter
parser.add_argument('--eval_interval', type=int, default=500)
parser.add_argument('--eval_iters', type=int, default=200)
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_save', type=str, default='trainedModel/Checkpoint')



test_type = 1 # 1 for model, 0 for dataset



args = parser.parse_args()

model = Graph2HeuristicModel(args)
filename = args.model_save + '_' + str(args.map_size) + '_' + str(235) + '.pth'
# Load the state dict back into the model
if test_type:
    model.load_state_dict(torch.load(filename))
    m = model.to(args.device)
    # Don't forget to set the model to evaluation mode if you're doing inference
    m.eval()
safec = 0.9
dim = 16

evaluatedataset = dataset_builder(dim, 'dataset/16risk/val/') #
evaluateDataLoader = DataLoader(evaluatedataset, batch_size=1, shuffle=True)
learnedexplored = 0
manhattanexplored = 0
learneddistance = 0
manhattandistance = 0
learnedwin = 0
manhattanwin = 0
noresult = 0
learnedtime = 0
manhattantime = 0
evaluateDataLoader = iter(evaluateDataLoader)
for i in tqdm(range(10000)):
    riskmap, start, dest, hmap = next(evaluateDataLoader)
    start2 = start.to(args.device)
    riskmap2 = riskmap.to(args.device)
    dest2 = dest.to(args.device)
    hmap2 = hmap.to(args.device)
    dest = dest.numpy()[0]
    start = start.numpy()[0]
    #transfer riskmap to dim*dim
    riskmap = np.array(riskmap)
    UAVmap = riskmap.reshape(dim, dim)
    #print(UAVmap.shape)
    #-----uncommand if model--------------------
    if test_type:
        logits, loss = model(riskmap2, start2, dest2, hmap2)
        logits = logits.to('cpu')
        logits = logits.detach().numpy()[0]
        hmap = logits.reshape(dim, dim)
    #-----------------------------------------
    else:
        hmap = hmap.numpy()[0].reshape(dim, dim)
    t0 = time.time()
    #print(logits)
    #if np.sum(hmap == -1) > dim*dim - 10:
        #continue
    #print(hmap)
    #print(dest)
    #print(UAVmap)
    
    #random pick xI
    xI = (start//dim, start%dim, 1)
    xG = (dest//dim, dest%dim)
    # while hmap[xI[0]][xI[1]] == -1:
    #     xI = (np.random.randint(0, dim), np.random.randint(0, dim), 1)
    #     xIindex = xI[0] * dim + xI[1]
    #learned hmap
    actionList, path, nodeList, count, explored = aStarSearch(xI, xG, UAVmap, safec, 'learning', hmap)
    if explored:
        learnedexplored += len(explored)
    if actionList:
        learneddistance += len(actionList)
    learnedtime += time.time() - t0
    
    t0 = time.time()
    actionList2, path2, nodeList2, count2, explored2 = aStarSearch(xI, xG, UAVmap, safec)
    if explored2:
        manhattanexplored += len(explored2)
    if actionList2:
        manhattandistance += len(actionList2)
    if actionList2:
        if len(actionList2) < len(actionList):
            manhattanwin += 1
        elif len(explored2) > len(explored):
            learnedwin += 1
        else:
            manhattanwin += 1
    elif actionList:
        print("something go wrong")
    else:
        noresult += 1
    manhattantime += time.time() - t0
total = learnedwin + manhattanwin + noresult
print("learnedexplored:", round(learnedexplored/total,2))
print("manhattanexplored:", round(manhattanexplored/total,2))
print("learneddistance:", round(learneddistance/total, 2))
print("manhattandistance:", round(manhattandistance/total, 2))
print("learnedwin:", round(learnedwin/total*100, 2))
print("manhattanwin:", round(manhattanwin/total*100, 2))
print("noresult:", noresult)
print("total: ", learnedwin + manhattanwin + noresult)
print("learnedtime:", round(learnedtime/total,4))
print("manhattantime:", round(manhattantime/total, 4))


