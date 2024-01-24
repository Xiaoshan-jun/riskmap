import argparse
from include.dataloader_large_data import dataset_builder
from torch.utils.data import DataLoader
from include.model import Graph2HeuristicModel
from tqdm import tqdm
import numpy as np
import torch

torch.manual_seed(1337)
batch_size = 64
map_size = 16
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def load_data(file, index, label_key, map_size=None, dtype=torch.float32):
    #help load data
    data = np.load(file)
    if label_key:
        data = data[f'data_{index}_{label_key}']
        if map_size:
            data = data[0] * map_size + data[1]
    return torch.tensor(data.reshape(-1), dtype=dtype)
def to_device(tensors, device):
    #help load data
    return [tensor.to(device) for tensor in tensors]
def estimate_loss(model, eval_iters, trainDataLoader, valDataLoader, args):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'train':
            data_iterator = iter(trainDataLoader)
        else:
            data_iterator = iter(valDataLoader)
        for k in range(eval_iters):
            graphfilenames, labelsfilenames, indexs = next(data_iterator)
            riskmap = []
            start = []
            destination = []
            hmap = []
            # load risk map
            for i in range(len(graphfilenames)):
                graphfilename = graphfilenames[i]
                rm = np.load(graphfilename)
                rm = np.array(rm).reshape(-1)
                riskmap.append(rm)
                # load labels
                labelsfilename = labelsfilenames[i]
                datapoints = np.load(labelsfilename)
                index = indexs[i]
                # load labels: start
                sta = datapoints[f'data_{index}_start']
                sta = sta[0] * args.map_size + sta[1]
                start.append(sta)
                # load labels: destination
                des = datapoints[f'data_{index}_destination']
                des = des[0] * args.map_size + des[1]
                destination.append(des)
                # load labels: expert heurisic
                hm = datapoints[f'data_{index}_Hvalue']
                hm = np.array(hm).reshape(-1)
                hmap.append(hm)
            riskmap = np.array(riskmap)
            riskmap = torch.tensor(riskmap, dtype=torch.float32)
            riskmap = riskmap.to(device)
            start = np.array(start)
            start = torch.tensor(start, dtype=torch.int)
            start = start.to(device)
            destination = np.array(destination)
            destination = torch.tensor(destination, dtype=torch.int)
            destination = destination.to(device)
            hmap = np.array(hmap)
            hmap = torch.tensor(hmap, dtype=torch.float32)
            hmap = hmap.to(device)
            logits, loss = model(riskmap, start, destination, hmap)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
def train():
    ##Dataset params
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--map_size', type=int, default= 16) #map is map_size * map_size
    parser.add_argument('--block_size', type=int, default= 16*16) #must equal to map_size**2 #block_size is the number of nodes
    #model parameter
    parser.add_argument('--n_embd', type=int, default= 864)  #embedded item size
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--n_head', type=int, default=6) #number of head, decide how many head the channels devided into.
    #train parameter
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--learning_iters', type=int, default=5000)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_save', type=str, default='trainedModel/Checkpoint')
    parser.add_argument('--continue_from_previous', type=bool, default=False)
    parser.add_argument('--previous_iter', type=str, default='465')


    args = parser.parse_args()
    print(torch.cuda.is_available())
    #
    print('loading train dataset')
    traindataset = dataset_builder(args.map_size, 'dataset/16risk/train/') #
    trainDataLoader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)
    #
    print('loading val dataset')
    valdataset = dataset_builder(args.map_size, 'dataset/16risk/val/')
    valDataLoader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=True)
    if args.continue_from_previous:
        print('load from previous')
        model = Graph2HeuristicModel(args)
        filename = args.model_save + '_' + str(args.map_size) + '_' + args.previous_iter + '.pth'
        model.load_state_dict(torch.load(filename))
        start = int(args.previous_iter)+1
    else:
        model = Graph2HeuristicModel(args)
        start = 0
    m = model.to(args.device)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=args.learning_rate)
    lowest = 10000
    for i in tqdm(range(start, args.max_iters)):
        if i % args.eval_interval == 0:
            losses = estimate_loss(m, args.eval_iters, trainDataLoader, valDataLoader, args)
            print(f"step{i}: train loss {losses['train']:.4f}. val loss {losses['val']:.4f}")
            if losses['val'] < lowest:
                lowest = losses['val']
                filename = args.model_save + '_' + str(args.map_size) + '_' + str(i) + '.pth'
                print('newlow found, saving the model: ', filename)
                torch.save(m.state_dict(), filename)
                filename = args.model_save + '_' + str(args.map_size) + '_' + str(i) + '_entire.pth'
                torch.save(m, filename)
        else:
            data_iterator = iter(trainDataLoader)
            for _ in range(args.learning_iters):
                graphfilenames, labelsfilenames, indexs = next(data_iterator)
                riskmap = []
                start = []
                destination = []
                hmap = []
                # load risk map
                for i in range(len(graphfilenames)):
                    graphfilename = graphfilenames[i]
                    rm = np.load(graphfilename)
                    rm = np.array(rm).reshape(-1)
                    riskmap.append(rm)
                    # load labels
                    labelsfilename = labelsfilenames[i]
                    datapoints = np.load(labelsfilename)
                    index = indexs[i]
                    # load labels: start
                    sta = datapoints[f'data_{index}_start']
                    sta = sta[0] * args.map_size + sta[1]
                    start.append(sta)
                    # load labels: destination
                    des = datapoints[f'data_{index}_destination']
                    des = des[0] * args.map_size + des[1]
                    destination.append(des)
                    # load labels: expert heurisic
                    hm = datapoints[f'data_{index}_Hvalue']
                    hm = np.array(hm).reshape(-1)
                    hmap.append(hm)
                riskmap = np.array(riskmap)
                riskmap = torch.tensor(riskmap, dtype=torch.float32)
                riskmap = riskmap.to(device)
                start = np.array(start)
                start = torch.tensor(start, dtype=torch.int)
                start = start.to(device)
                destination = np.array(destination)
                destination = torch.tensor(destination, dtype=torch.int)
                destination = destination.to(device)
                hmap = np.array(hmap)
                hmap = torch.tensor(hmap, dtype=torch.float32)
                hmap = hmap.to(device)
                logits, loss = m(riskmap, start, destination, hmap)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

if __name__=='__main__':
    train()