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



test_type = 0 # 1 for model, 0 for dataset



args = parser.parse_args()

model = Graph2HeuristicModel(args)
filename = args.model_save + '_' + str(args.map_size) + '_' + str(10) + '.pth'
# Load the state dict back into the model
if test_type:
    model.load_state_dict(torch.load(filename))
    m = model.to(args.device)
    # Don't forget to set the model to evaluation mode if you're doing inference
    m.eval()
safec = 0.8
dim = 16

evaluatedataset = dataloader(dim, 'dataset/16wind/test/') #
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
for i in range(200):
    riskmap, start, dest, hmap = next(evaluateDataLoader)
    start2 = start.to(args.device)
    riskmap2 = riskmap.to(args.device)
    dest2 = dest.to(args.device)
    hmap2 = hmap.to(args.device)
    #logits, loss = model(riskmap2, start2, dest2, hmap2)
    dest = dest.numpy()[0]
    start = start.numpy()[0]
    #transfer riskmap to dim*dim
    riskmap = np.array(riskmap)
    UAVmap = riskmap.reshape(dim, dim)
    #print(UAVmap.shape)
    #logits = logits.to('cpu')
    #logits = logits.detach().numpy()[0]
    #hmap = logits.reshape(dim, dim)
    t0 = time.time()
    hmap = hmap.numpy()[0].reshape(dim, dim)
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
    print(xI)
    print(xG)
    print(path)
    learnedtime += time.time() - t0
    cmap = plt.cm.gray_r
    cmap.set_under('white')  # values below the lowest in the colormap are white
    cmap.set_over('black')   # values above the highest in the colormap are black
    fig, ax = plt.subplots()
    cax = ax.imshow(UAVmap, cmap=cmap, vmin=0, vmax=0.2)
        # Plotting the matrix with colors defined by the numbers
    #plt.imshow(UAVmap, cmap='gray_r')
    fig.colorbar(cax) # Adds a color bar to match the color scale
    plt.title('wind flow map case example2')
    plt.xticks(np.arange(UAVmap.shape[1]), np.arange(UAVmap.shape[1]))
    plt.yticks(np.arange(UAVmap.shape[0]), np.arange(UAVmap.shape[0]))
    hd_image_path = 'figure/randommapexample' + str(i) + '.png'
    fig.savefig(hd_image_path, dpi=900)
    #plot path
    ax.add_patch(plt.Rectangle((xI[1]-0.5, xI[0]-0.5), 1, 1, edgecolor='yellow', facecolor='none', lw=2))
    for (x, y, s) in path:
        ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, edgecolor='red', facecolor='none', lw=2))
    ax.add_patch(plt.Rectangle((xG[1]-0.5, xG[0]-0.5), 1, 1, edgecolor='green', facecolor='none', lw=2))
    matrix = np.zeros((dim,dim))
    #plot explored
    for (x, y, s) in explored:
        matrix[x][y] += 1
    for x in range(dim):
        for y in range(dim):
            if matrix[x][y] > 0:
                ax.text(y, x, f'{int(matrix[x, y])}', ha='center', va='center', color='blue')
    plt.title('path founded by A* with learned heuristic')            
    hd_image_path = 'figure/randommapexampleleanred'+ str(i) + '.png'
    fig.savefig(hd_image_path, dpi=900)
    plt.show()
    if explored:
        learnedexplored += len(explored)
    if actionList:
        learneddistance += len(actionList)
    
    t0 = time.time()
    actionList2, path2, nodeList2, count2, explored2 = aStarSearch(xI, xG, UAVmap, safec)
    cmap = plt.cm.gray_r
    cmap.set_under('white')  # values below the lowest in the colormap are white
    cmap.set_over('black')   # values above the highest in the colormap are black
    fig, ax = plt.subplots()
    cax = ax.imshow(UAVmap, cmap=cmap, vmin=0, vmax=0.2)
        # Plotting the matrix with colors defined by the numbers
    #plt.imshow(UAVmap, cmap='gray_r')
    fig.colorbar(cax) # Adds a color bar to match the color scale
    plt.xticks(np.arange(UAVmap.shape[1]), np.arange(UAVmap.shape[1]))
    plt.yticks(np.arange(UAVmap.shape[0]), np.arange(UAVmap.shape[0]))
    hd_image_path = 'figure/randommapexample.png'
    fig.savefig(hd_image_path, dpi=900)
    #plot path
    ax.add_patch(plt.Rectangle((xI[1]-0.5, xI[0]-0.5), 1, 1, edgecolor='yellow', facecolor='none', lw=2))
    for (x, y, s) in path2:
        ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, edgecolor='red', facecolor='none', lw=2))
    ax.add_patch(plt.Rectangle((xG[1]-0.5, xG[0]-0.5), 1, 1, edgecolor='green', facecolor='none', lw=2))
    matrix = np.zeros((dim,dim))
    #plot explored
    for (x, y, s) in explored2:
        matrix[x][y] += 1
    for x in range(dim):
        for y in range(dim):
            if matrix[x][y] > 0:
                ax.text(y, x, f'{int(matrix[x, y])}', ha='center', va='center', color='blue')
    plt.title('path founded by A* with manhattan heuristic')            
    hd_image_path = 'figure/randommapexampleManhattan'+ str(i) + '.png'
    fig.savefig(hd_image_path, dpi=900)
    plt.show()
    if explored2:
        manhattanexplored += len(explored2)
    if actionList2:
        manhattandistance += len(actionList2)
    if actionList2:
        if len(actionList2) > len(actionList):
            manhattanwin += 1
        elif len(explored2) > len(explored):
            learnedwin += 1
        else:
            learnedwin += 1
    elif actionList:
        print("something go wrong")
    else:
        noresult += 1
    manhattantime += time.time() - t0 
print("learnedexplored:", learnedexplored)
print("manhattanexplored:", manhattanexplored)
print("learneddistance:", learneddistance)
print("manhattandistance:", manhattandistance)
print("learnedwin:", learnedwin)
print("manhattanwin:", manhattanwin)
print("noresult:", noresult)
print("total: ", learnedwin + manhattanwin + noresult)
print("learnedtime:", learnedtime)
print("manhattantime:", manhattantime)