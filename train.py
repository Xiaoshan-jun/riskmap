import argparse
from include.dataloader import dataloader
from torch.utils.data import DataLoader
from include.model import Graph2HeuristicModel
from tqdm import tqdm
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

def estimate_loss(model, eval_iters, trainDataLoader, valDataLoader):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'train':
            data_iterator = iter(trainDataLoader)
        else:
            data_iterator = iter(valDataLoader)
        for k in range(eval_iters):
            riskmap, start, dest, hmap = next(data_iterator)
            riskmap = riskmap.to(device)
            start = start.to(device)
            dest = dest.to(device)
            hmap = hmap.to(device)
            logits, loss = model(riskmap, start, dest, hmap)
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
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_save', type=str, default='trainedModel/Checkpoint')
    args = parser.parse_args()
    print(torch.cuda.is_available())
    #
    print('loading train dataset')
    traindataset = dataloader(args.map_size, 'dataset/16wind/train/') #
    trainDataLoader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)
    #
    print('loading val dataset')
    valdataset = dataloader(args.map_size, 'dataset/16wind/val/') 
    valDataLoader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=True)
    model = Graph2HeuristicModel(args)
    m = model.to(args.device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=args.learning_rate)
    lowest = 10000
    for iter in tqdm(range(args.max_iters)):
        if iter % args.eval_interval == 0:
            losses = estimate_loss(m, args.eval_iters, trainDataLoader, valDataLoader)
            print(f"step{iter}: train loss {losses['train']:.4f}. val loss {losses['val']:.4f}")
            if losses['val'] < lowest:
                lowest = losses['val']
                filename = args.model_save + '_' + str(args.map_size) + '_' + str(iter) + '.pth'
                print('newlow found, saving the model: ', filename)
                torch.save(m.state_dict(), filename)
                filename = args.model_save + '_' + str(args.map_size) + '_' + str(iter) + '_entire.pth'
                torch.save(m, filename)
        else:
            for riskmap, start, dest, hmap in trainDataLoader:
                riskmap = riskmap.to(device)
                start = start.to(device)
                dest = dest.to(device)
                hmap = hmap.to(device)
                logits, loss = m(riskmap, start, dest, hmap)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

if __name__=='__main__':
    train()