import argparse
from include.dataloader import dataloader
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
def manhattanHeuristic(state, goal):
   """ewg, newc)
   A heuristic function estimates the cost from the current state to the nearest
   goal.  This heuristic is trivial.
   """
   return abs(goal[0] - state[0]) + abs(goal[1] - state[1])
def test():
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
    args = parser.parse_args()

    model = Graph2HeuristicModel(args)
    filename = args.model_save + '_' + str(args.map_size) + '_' + str(1000) + '.pth'
    # Load the state dict back into the model
    model.load_state_dict(torch.load(filename))
    m = model.to(args.device)
    # Don't forget to set the model to evaluation mode if you're doing inference
    m.eval()
    safec = 0.9
    dim = 16
    evaluatedataset = dataloader(args.map_size, 2) #2 = evaluate, 1 = train, 0 = test
    evaluateDataLoader = DataLoader(evaluatedataset, batch_size=1, shuffle=True)
    learnedexplored = 0
    manhattanexplored = 0
    learneddistance = 0
    manhattandistance = 0
    learnedwin = 0
    manhattanwin = 0
    for i in tqdm(range(100)):
        data_iterator = iter(evaluateDataLoader)
        riskmap, dest, hmap = next(data_iterator)
        riskmap2 = riskmap.to(args.device)
        dest2 = dest.to(args.device)
        hmap2 = hmap.to(args.device)
        logits, loss = model(riskmap2, dest2, hmap2)
        dest = int(dest)
        #transfer riskmap to dim*dim
        riskmap = np.array(riskmap)
        UAVmap = riskmap.reshape(dim, dim)
        #print(UAVmap.shape)
        logits = logits.to('cpu')
        logits = logits.detach().numpy()[0]
        hmap = hmap.detach().numpy()[0]
        #print(logits)
        if np.sum(hmap == -1) > dim*dim - 10:
            continue
        print(hmap)
        print(logits)
        print(loss)
        #random pick xI
        xI = (np.random.randint(0, dim), np.random.randint(0, dim), 1)
        xIindex = xI[0] * dim + xI[1]
        xG = (dest//dim, dest%dim)
        #print(xG)
        while hmap[xIindex] == -1:
            xI = (np.random.randint(0, dim), np.random.randint(0, dim), 1)
            xIindex = xI[0] * dim + xI[1]

        #learned hmap
        actionList, path, nodeList, count, explored = aStarSearch(xI, xG, UAVmap, safec, 'learning', hmap)
        if explored:
            learnedexplored += len(explored)
        if actionList:
            learneddistance += len(actionList)



        actionList2, path2, nodeList2, count2, explored2 = aStarSearch(xI, xG, UAVmap, safec)
        if explored2:
            manhattanexplored += len(explored2)
        if actionList2:
            manhattandistance += len(actionList2)
        if actionList2:
            if actionList2 > actionList:
                manhattanwin += 1
            elif explored2 > explored:
                learnedwin += 1
            else:
                manhattanwin += 1
        
    print("learnedexplored:", learnedexplored)
    print("manhattanexplored:", manhattanexplored)
    print("learneddistance:", learneddistance)
    print("manhattandistance:", manhattandistance)
    print("learnedwin:", learnedwin)
    print("manhattanwin:", manhattanwin)

if __name__=='__main__':
    test()
