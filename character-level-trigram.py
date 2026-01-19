import torch
import torch.nn.functional as F

# -----------------------------
# 1. Load data
# -----------------------------
words = open('names.txt', 'r').read().splitlines()

# Build character mappings
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}  # start at 1
stoi['.'] = 0  # special start/end token
itos = {i:s for s,i in stoi.items()}
vocab_size = len(stoi)

# -----------------------------
# 2. Create dataset (trigrams)
# -----------------------------
xs, ys = [], []

for w in words:
    chs = ['.', '.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append([ix1,ix2])
        ys.append(ix3)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.shape[0]

print(f"Number of examples: {num_examples}")

# -----------------------------
# 3. Initialize model
# -----------------------------
torch.manual_seed(2147483647)
W = torch.randn((vocab_size*2, vocab_size), requires_grad=True)

# -----------------------------
# 4. Training loop (gradient descent)
# -----------------------------
lr = 10  # learning rate
reg = 0.1  # L2 regularization strength
steps = 500

for step in range(steps):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=vocab_size).float()  # one-hot input
    xenc = xenc.view(num_examples, vocab_size*2)
    logits = xenc @ W                                    # log-counts
    counts = logits.exp()                                 # counts (unnormalized)
    probs = counts / counts.sum(1, keepdims=True)        # softmax probabilities
    
    # Loss: negative log-likelihood + L2 regularization
    loss = -probs[torch.arange(num_examples), ys].log().mean() + reg * (W**2).mean()
    
    # Backward pass
    W.grad = None
    loss.backward()
    
    # Gradient descent update
    W.data -= lr * W.grad
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# -----------------------------
# 5. Sampling from the trained model
# -----------------------------
g = torch.Generator().manual_seed(2147483647)

g = torch.Generator().manual_seed(2147483647)

for _ in range(10):
    out = []
    xprev = [0, 0]  # start tokens (two of them)
    while True:
        xenc = F.one_hot(torch.tensor([xprev]), num_classes=vocab_size).float()
        xenc = xenc.view(1, vocab_size*2)
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        xprev = [xprev[1], ix]  # shift window
        if ix == 0:
            break
    print("".join(out))

