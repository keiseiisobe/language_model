'''
The first way of training bigram character-level language model.
We count the frequency of a set of characters and normalize it to get probability distribution.
'''

import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("names.txt", 'r') as f:
        lines = f.read().splitlines()
    parameter = torch.zeros((27, 27), dtype=torch.int32) # 27 = len(a ~ z + .)
    chars = sorted(list(set(''.join(lines))))
    stoi = { s:i for i, s in enumerate(chars) }
    stoi["."] = 26
    itos = { i:s for s, i in stoi.items() }

    # training
    for line in lines:
        line = ["."] + list(line) + ["."]
        for x, y in zip(line, line[1:]):
            parameter[stoi[x], stoi[y]] += 1

    # plot distribution
    fig, axs = plt.subplots(1, 2, figsize=(16, 16))
    axs[0].imshow(parameter, cmap="Blues")
    for i in range(27):
        for j in range(27):
            char_set = itos[i] + itos[j]
            axs[0].text(j, i, char_set, fontsize=5, ha="center", va="bottom")
            axs[0].text(j, i, parameter[i, j].item(), fontsize=5, ha="center", va="top")
    axs[0].set_title("Frequency of a set of two characters")

    parameter = parameter.to(torch.float32)
    parameter /= parameter.sum(dim=1, keepdim=True)
    axs[1].imshow(parameter, cmap="Blues")
    for i in range(27):
        for j in range(27):
            char_set = itos[i] + itos[j]
            axs[1].text(j, i, char_set, fontsize=5, ha="center", va="bottom")
            axs[1].text(j, i, round(parameter[i, j].item(), ndigits=2), fontsize=5, ha="center", va="top")
    axs[1].set_title("Probability distribution per a raw")
    plt.show()

    # test
    print(f"test{'-'*50}")
    for i in range(10):
        raw = 26 # itos[26] = "."
        out = []
        while True:
            p = parameter[raw]
            raw = p.multinomial(1).item()
            out.append(itos[raw])
            if raw == 26:
                break
        print(''.join(out))

    # evaluation
    log_likelyhood = 0.0
    n = 0
    for line in lines[:3]:
        line = ["."] + list(line) + ["."]
        for x, y in zip(line, line[1:]):
            prob = parameter[stoi[x], stoi[y]]
            log_likelyhood += torch.log(prob)
            n += 1
    print(f"evaluation{'-'*50}")
    print(f"log likelyhood: {log_likelyhood}")
    print(f"negative log likelyhood: {-log_likelyhood}")
    print(f"average negative log likelyhood: {-log_likelyhood / n}")
