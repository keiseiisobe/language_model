import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.w = torch.randn((fan_in, fan_out)) * (1 / fan_in**0.5)
        self.b = torch.zeros((1, fan_out)) if bias else None

    def __call__(self, x):
        return x @ self.w + (self.b if self.b is not None else 0)

    def parameters(self):
        return [self.w] + ([self.b] if self.b is not None else [])

class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var =  torch.ones(dim)
        self.training = True

    def __call__(self, x):
        """
        see Algorithm1 in https://arxiv.org/pdf/1502.03167
        """
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var
        normalized = x - mean / torch.sqrt(var + self.eps)
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        return self.gamma * normalized + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.tanh(x)

    def parameters(self):
        return []

class Embedding:
    def __init__(self, n_sample, n_emb):
        self.w = torch.randn((n_sample, n_emb))

    def __call__(self, x):
        return self.w[x]

    def parameters(self):
        return [self.w]

class Flatten:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.view(x.shape[0], -1)

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    with open("names.txt", "r") as f:
        lines = f.read().splitlines()
    chars = sorted(set(list(''.join(lines))))
    stoi = { s:i+1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = { i:s for s, i in stoi.items() }

    n_blocks = 3
    def build_dataset(lines):
        X, Y = [], []
        for line in lines:
            context = [0] * n_blocks
            for c in line + ".":
                X.append(context)
                Y.append(stoi[c])
                context = context[1:] + [stoi[c]]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    random.shuffle(lines)
    # Basically, training set: 80%, validation set: 10%, test set: 10%
    trset_len = int(len(lines) * 0.8)
    vaset_len = int(len(lines) * 0.9)
    X_tr, Y_tr = build_dataset(lines[:trset_len])
    X_va, Y_va = build_dataset(lines[trset_len:vaset_len])
    X_te, Y_te = build_dataset(lines[vaset_len:])

    n_emb = 10
    n_sample = len(itos)
    n_hidden = 200
    model = Sequential([
        Embedding(n_sample, n_emb),
        Flatten(),
        Linear(n_blocks * n_emb, n_hidden, bias=False),
        BatchNorm(n_hidden),
        Tanh(),
        Linear(n_hidden, n_sample)
    ])

    for parameter in model.parameters():
        parameter.requires_grad = True

    # training
    batch_size = 32
    learning_rate = 0.1
    epochs = 10000
    losses = []
    for i in range(epochs):
        indices = torch.randint(0, len(X_tr), (batch_size,))
        X, Y = X_tr[indices], Y_tr[indices]
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        losses.append(loss)
        for parameter in model.parameters():
            parameter.grad = None
        loss.backward()
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad
    plt.plot(torch.tensor(losses).view(-1, 100).mean(dim=1))
    plt.show()

    for layer in model.layers:
        layer.training = False
        
    # validation, test
    @torch.no_grad()
    def split_loss(split):
        x, y = {
            "train": (X_tr, Y_tr),
            "val": (X_va, Y_va),
            "test": (X_te, Y_te)
        }[split]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        print(f"{split}_loss: {loss}")

    split_loss("train")
    split_loss("val")

    # sample from trained distribution
    for _ in range(10):
        x = [0] * n_blocks
        i = 0
        out = []
        while True:
            logits = model(torch.tensor([x]))
            probs = F.softmax(logits, dim=1)
            i = probs.multinomial(1).item()
            out.append(itos[i])
            x = x[1:] + [i]
            if i == 0:
                break
        print(''.join(out))
    
