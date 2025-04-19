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
            #print(''.join(itos[i] for i in context), "---->", char)
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

    # create embedding vector (2 dimentional)
    # first layer
    C = torch.randn((27, 2), requires_grad=True)
    # is it possible to create matrix directly by using torch.randn((32, 3, 2)) ?
    
    # second layer
    # You can see underfitting if you change 300 -> 100 parameters
    W1 = torch.randn((6, 300), requires_grad=True)
    b1= torch.randn((1, 300), requires_grad=True)

    # third layer
    W2 = torch.randn((300, 27), requires_grad=True)
    b2 = torch.randn((1, 27), requires_grad=True)

    parameters = [C, W1, b1, W2, b2]

    # training
    print(f"training{'-'*50}")
    epochs = 100000
    # learning rate decay
    # lre = torch.linspace(-3, 0, epochs)
    # lre = lre.flip(dims=(0,))
    # lrs = 10**lre
    losses = []
    for epoch in range(epochs):
        i = torch.randint(0, X_tr.shape[0], (32,))
        h = torch.tanh(C[X_tr[i]].view((-1, 6)) @ W1 + b1)
        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, y_tr[i])
        losses.append(loss.item())
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data -= 0.1 * p.grad
    print(f"loss while training: {loss}")

    plt.plot(losses)
    plt.title("Loss in training set")
    plt.show()

    # evaluation
    print(f"evaluation{'-'*50}")
    i = torch.randint(0, X_ev.shape[0], (32,))
    h = torch.tanh(C[X_ev[i]].view((-1, 6)) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y_ev[i])
    print(f"loss while evaluation: {loss}")

    for i in range(10):
        x = [stoi["."]] * n_blocks
        raw = 26 # itos[26] = "."
        out = []
        while True:
            h = torch.tanh(C[x].view((-1, 6)) @ W1 + b1)
            logits = h @ W2 + b2
            softmax = nn.Softmax(dim=1)
            probs = softmax(logits)
            raw = probs.multinomial(1).item()
            out.append(itos[raw])
            x = x[1:] + [raw]
            if raw == 26:
                break
        print(''.join(out))
    
