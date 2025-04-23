import sys
sys.path.append('/home/kisobe/sgoinfre/pip')

from os import path
import torch
import gpt

if __name__ == "__main__":
    # hyperparameters
    n_blocks = 256
    emb_dim = 384
    n_heads = 6
    dropout = 0.2

    # get data
    input_file_path = path.dirname(__file__) + "/input.txt"
    with open(input_file_path, "r") as f:
        data = f.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = { s:i for i, s in enumerate(chars) }
    itos = { i:s for i, s in enumerate(chars) }
    encoder = lambda chars: [stoi[c] for c in chars]
    decoder = lambda ints: [itos[i] for i in ints]

    model = gpt.BigramLanguageModel(vocab_size, emb_dim, n_blocks, n_heads, dropout)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.generate(torch.zeros((1, 1), dtype=torch.int32), 1000, n_blocks, decoder)
