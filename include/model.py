import torch
import torch.nn as nn
from torch.nn import functional as F
"""attention
there are three linear layers, each take (B, T, C) and output (B, T, head_size).
we called there outputs K, Q ,v.
Then we use q@k^t, divided C**0.5 to get weights.
we softmax and dropout the weights.
Finally, the output of head layer will be weight @ v. THe output size is (B, T, head_size)
"""
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, C) -> (B, T, head_size)
        q = self.query(x) #(B, T, C) -> -> (B, T, head_size)
        wei  = q @ k.transpose(-2, -1) * C**-0.5 #(B, T, head_size) @ (B, head_size, T) -> (B, T, T) normalized
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T) masked
        wei = F.softmax(wei, dim=-1) #(B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v #(B, T, T) @ (B, T, C) ->(B, T, head_size)
        return out

"""MultiHeadAttention
The input(B, T, C) will be send to multiple(n_head) head layers. the size of each head layer will be (B, T, headsize), where headsize =  head_size = C // n_head
then the output of all the head are cat together, form (B, T, C)
We project and dropout the output
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #projection for residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #cat over the channel dimension e.g(B, T, 1/2C) cat (b, t, 1/2C) num_heads = 2 FINNALly get (B, T, C)
        out = self.dropout(self.proj(out))
        return out

    """a simple linear layer followed by a non-linearity
    
    """
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn. Linear(n_embd, 4 * n_embd),
                                 nn.ReLU(),
                                 nn. Linear(4 * n_embd, n_embd), #projection going back to residual pathway
                                 nn.Dropout(dropout),
                                 )

    def forward(self, x):
        return self.net(x)

"""
transformer block
the input will be send to multihead attention layer with residual connection, then be send to a FeedForward layer with residual connection for non-linearity
"""
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'de like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) #layer normalization, trainable

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #residual connection (B, T, C)
        x = x + self.ffwd(self.ln2(x)) #residual connection (B, T, C)
        return x
"""

customer loss = MSE + overestimate 
"""
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.01):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_target):
        mse_loss = torch.mean((y_pred - y_target) ** 2)
        penalty = torch.mean(nn.functional.relu(y_pred - y_target + self.beta))

        return mse_loss + self.alpha * penalty

"""the main model 
the input is x*x risky and (1, 1) destination, the output is x*x heuristic
heuristic size is x*x + 1
we embeded there input into (B, T, C), T is the number of nodes, C is the size of embed. 
we add them together, then send to the transfromer attention layer.
"""
class Graph2HeuristicModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.risk_input = nn.Linear(args.block_size, args.block_size * args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd) #block_size is the number of nodes
        self.destination_embedding_table = nn.Embedding(args.block_size, args.n_embd) #
        self.blocks = nn.Sequential(
            *[Block(args.n_embd, args.n_head, args.dropout) for _ in range(args.n_layer)]
        )
        self.ln_f = nn.LayerNorm(args.n_embd)  # final layer norm
        self.lm_head = nn.Linear(args.n_embd, args.block_size)  # output layer
        self.loss_fn = CustomLoss(alpha=1.0, beta=0.01)
        self.device = args.device


    def forward(self, risk, destination, targets = None):
        B, T = risk.shape #B is batch, T is the number of nodes, C is the channel of of each number of nodes
        #idx is (1, x*x), destination is (1, 1)
        pos_emb = self.position_embedding_table(torch.arange(T, device = self.device)) #(T, C)
        des_emb = self.destination_embedding_table(destination) #(B, C)
        risk_emb = self.risk_input(risk) #(B, T*C)
        #print(pos_emb.shape)
        #print(des_emb.shape)
        #print(risk_emb.shape)
        des_emb = des_emb.unsqueeze(1).expand(risk_emb.size(0), pos_emb.size(0), pos_emb.size(1))#(B, T, C)
        #print(des_emb.shape)
        risk_emb = risk_emb.view(risk_emb.size(0), pos_emb.size(0), pos_emb.size(1))#(B, T, C)
        #print(risk_emb.shape)
        x = pos_emb + des_emb + risk_emb#(B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x)#(B, T, T)
        #print(logits.shape)
        ans = logits.mean(dim=2)#(B, T)
        #print(ans.shape)
        loss = self.loss_fn(ans, targets)

        return logits, loss

    def generate(self, idx):
        logits, loss = self(idx)
