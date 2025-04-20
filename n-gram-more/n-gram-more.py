import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open("names.txt", 'r') as f:
        lines = f.read().splitlines()
    chars = sorted(list(set(''.join(lines))))
    stoi = { s:i for i, s in enumerate(chars) }
    stoi["."] = 26
    itos = { i:s for s, i in stoi.items() }

    n_blocks = 3
    X = []
    y = []
    for line in lines:
        line = line + "."
        context = [stoi["."]] * n_blocks
        for char in line:
            X.append(context)
            y.append(stoi[char])
            context = context[1:] + [stoi[char]]
    X = torch.tensor(X)
    y = torch.tensor(y)
    tr_ind = int(len(X) * 0.8)
    ev_ind = tr_ind + int(len(X) * 0.1)
    X_tr = X[:tr_ind]
    X_ev = X[tr_ind:ev_ind]
    X_te = X[ev_ind:]
    y_tr = y[:tr_ind]
    y_ev = y[tr_ind:ev_ind]
    y_te = y[ev_ind:]

    n_emb = 10
    n_hidden = 200
    C = torch.randn((27, n_emb))
    W1 = torch.randn((n_emb * n_blocks, n_hidden)) * (5/3) / ((n_emb * n_blocks)**0.5) # kaiming initialization
    b1 = torch.randn((1, n_hidden)) * 0
    W2 = torch.randn((n_hidden, 27)) * 0.1
    b2 = torch.randn((1, 27)) * 0

    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    
    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for parameter in parameters:
        parameter.requires_grad = True

    # training
    print(f"training{'-'*50}")
    epochs = 10000
    batch_size = 32
    losses = []
    for epoch in range(epochs):
        i = torch.randint(0, X_tr.shape[0], (batch_size,))
        emb = C[X_tr[i]]
        preh = emb.view(emb.shape[0], -1) @ W1 + b1
        # batch normalization
        preh = bngain * (preh - preh.mean(0, keepdim=True)) / preh.std(0, keepdim=True) + bnbias
        h = torch.tanh(preh)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, y_tr[i])
        losses.append(loss.item())
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data -= 0.1 * p.grad
        if epoch % 1000 == 0:
            print(f"loss while training: {loss}")

    plt.plot(losses)
    plt.title("Loss in training set")
    plt.show()

    # evaluation
    print(f"evaluation{'-'*50}")
    i = torch.randint(0, X_ev.shape[0], (32,))
    emb = C[X_ev[i]]
    preh = emb.view(emb.shape[0], -1) @ W1 + b1
    # batch normalization
    preh = bngain * (preh - preh.mean(0, keepdim=True)) / preh.std(0, keepdim=True) + bnbias
    h = torch.tanh(preh)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y_ev[i])
    print(f"loss while evaluation: {loss}")

    for i in range(10):
        x = [stoi["."]] * n_blocks
        raw = 26 # itos[26] = "."
        out = []
        while True:
            h = torch.tanh(C[x].view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            softmax = nn.Softmax(dim=1)
            probs = softmax(logits)
            raw = probs.multinomial(1).item()
            out.append(itos[raw])
            x = x[1:] + [raw]
            if raw == 26:
                break
        print(''.join(out))
    
