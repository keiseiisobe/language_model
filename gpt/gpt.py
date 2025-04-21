from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

class BigramLanguageModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.embedding = nn.Embedding(n, n)

    def forward(self, x, y=None):
        logits = self.embedding(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, -1)
        if y is None:
            loss = None
        else:
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self):
        x = torch.tensor([[0]])
        out = []
        for _ in range(300):
            logits, loss = self(x)
            probs = F.softmax(logits, dim=1)
            x_next = torch.multinomial(probs, 1)
            out.append(x_next.item())
            x.data = x_next.data
        return out
        

if __name__ == "__main__":
    input_file_path = path.dirname(__file__) + "/input.txt"
    if not path.exists(input_file_path):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        r = requests.get(url)
        with open(input_file_path, "w") as f:
            f.write(r.text)

    with open(input_file_path, "r") as f:
        data = f.read()

        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        stoi = { s:i for i, s in enumerate(chars) }
        itos = { i:s for i, s in enumerate(chars) }

        encoder = lambda chars: [stoi[c] for c in chars]
        decoder = lambda ints: [itos[i] for i in ints]

        data = torch.tensor(encoder(data))
        trset_len = int(len(data) * 0.9)
        train_data = data[:trset_len]
        val_data = data[trset_len:]

        n_blocks = 8
        batch_size = 4

        def get_batch(split):
            data = train_data if split == "train" else val_data
            indices = torch.randint(0, len(data) - n_blocks, (batch_size,))
            X = torch.stack([data[i:i+n_blocks] for i in indices])
            Y = torch.stack([data[i+1:i+n_blocks+1] for i in indices])
            return X, Y

        X, Y = get_batch("train")
        bigram = BigramLanguageModel(vocab_size)

        # training
        epochs = 10000
        optimizer = torch.optim.AdamW(bigram.parameters(), lr=1e-3)
        for epoch in range(epochs):
            logits, loss = bigram(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"loss: {loss}")
        out = decoder(bigram.generate())
        print(''.join(out))
