'''
The second way of training bigram character-level language model.
We optimize parameters in our neural network.
'''
import torch
from torch import nn
import torch.nn.functional as F

if __name__ == '__main__':
    with open("names.txt", 'r') as f:
        lines = f.read().splitlines()
    parameter = torch.zeros((27, 27), dtype=torch.int32) # 27 = len(a ~ z + .)
    chars = sorted(list(set(''.join(lines))))
    stoi = { s:i for i, s in enumerate(chars) }
    stoi["."] = 26
    itos = { i:s for s, i in stoi.items() }

    # training
    print(f"training{'-'*50}")
    xs, ys = [], []
    for line in lines:
        line = ["."] + list(line) + ["."]
        for x, y in zip(line, line[1:]):
            xs.append(stoi[x])
            ys.append(stoi[y])
    xs = torch.tensor(xs)
    xs = F.one_hot(xs, num_classes=27).float()
    ys = torch.tensor(ys)
    W = torch.randn((27, 27), requires_grad=True)

    for i in range(100):
        logits = xs @ W
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        loss = -probs[torch.arange(probs.shape[0]), ys].log().mean() # average negative log likelyhood
        print(f"loss: {loss}")
        W.grad = None
        loss.backward()
        W.data -= 50 * W.grad

    # test
    print(f"test{'-'*50}")
    for i in range(10):
        raw = 26 # itos[26] = "."
        out = []
        while True:
            x = F.one_hot(torch.tensor([raw]), num_classes=27).float()
            logits = x @ W
            probs = softmax(logits)
            raw = probs.multinomial(1).item()
            out.append(itos[raw])
            if raw == 26:
                break
        print(''.join(out))
