# Bigram character-level language model

implement a bigram character-level language model inspired by [this course](https://github.com/karpathy/makemore).

There are two ways.

## bigram_1.py
1. count the frequency of a set of characters (left figure)
2. normalize it and get probability distribution (right figure)
![](/bigram/Figure_1.png)

## bigram_2.py
1. make datasets(inputs, labels).
2. train through the neural network (you can see improvement of parameters).

```bash
training--------------------------------------------------
loss: 3.674828290939331
loss: 3.319348096847534
loss: 3.1123571395874023
loss: 2.979130983352661
...
loss: 2.4733636379241943
loss: 2.4731311798095703
loss: 2.4729034900665283
loss: 2.4726808071136475
test--------------------------------------------------
kon.
mfvililovila.
xw.
maikeikadahenie.
ezaur.
tde.
aitan.
ely.
rine.
kalahamessiy.

```
