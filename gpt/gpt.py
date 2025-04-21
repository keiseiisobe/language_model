from os import path
import torch
import requests

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

trset_len = len(data) * 0.9

n_blocks = 8
