import torch
from torch import nn as nn
from torch import optim as optim
import numpy as np
from collections import Counter
import random

#step 1: prepare the data
text = "the quick brown fox jumps over the lazy dog"
words = text.split()
vocab = set(words)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# print(word_to_idx)
# print()
# print(idx_to_word)

# step 2: generate skip-gram pairs
def generate_training_data(words, window_size):
    pass