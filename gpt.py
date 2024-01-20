import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2024)

# hyperparameters
batch_size = 64
block_size = 256
n_embd = 384
n_heads = 6
n_blocks = 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
device = "cuda:7" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

# read txt
with open("notebooks/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# encode/decode
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        weight = q @ k.transpose(-2, -1)  # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)  # randomly drop some communications

        v = self.value(x)  # (B, T, H)
        out = weight @ v  # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.project = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.project(self.dropout(out))
        return out

class MultiHeadAttentionParallel(nn.Module):
    def __init__(self, head_size, n_heads):
        super().__init__()
        self.linear_qkv = nn.Linear(n_embd, 3 * head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.project = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.head_size = head_size
    
    def forward(self, x):
        B, T, C = x.shape 
        q, k, v = self.linear_qkv(x).split(self.head_size, dim=2)
        q = q.view(B, T, self.n_heads, self.head_size // self.n_heads).transpose(2, 1) # (B, nh, T, h)
        k = k.view(B, T, self.n_heads, self.head_size // self.n_heads).transpose(2, 1)
        v = v.view(B, T, self.n_heads, self.head_size // self.n_heads).transpose(2, 1)

        weight = q @ k.transpose(-2, -1) # (B, nh, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim = -1)
        weight = self.attn_dropout(weight)

        out = weight @ v # (B, nh, T, T) @ (B, nh, T, h) = (B, nh, T, h)
        out = out.transpose(1, 2).contiguous().view(B, T, self.head_size) # (B, T, H)
        out = self.project(self.proj_dropout(out)) 
        return out 
        

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):
    def __init__(self, head_size, n_heads):
        super().__init__()
        #self.mhsa = MultiHeadAttention(head_size // n_heads, n_heads)  # communication
        self.mhsa = MultiHeadAttentionParallel(head_size, n_heads)
        self.ffwd = FeedForward(n_embd)  # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanuguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_blocks)])
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, H) data-dependent weight (self-attention mechanism)
        x = self.ffwd(self.ln(x))  # data-independent weight. Per node calculation.
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, -1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanuguageModel().to(device)
warmup_steps = 300
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
lambda_lr = (
    lambda iter: 0.2
    * n_embd**-0.5
    * min((iter + 1) ** -0.5, (iter + 1) * warmup_steps**-1.5)
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

print(f"number of parameters = {sum(p.numel() for p in model.parameters())}")

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        loss_est = estimate_loss()
        print(
            f"step {iter}/{max_iters}: train_loss {loss_est['train']:.4f}, valid_loss {loss_est['val']:.4f}, "
            f"learning rate = {optimizer.param_groups[0]['lr']:e}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))