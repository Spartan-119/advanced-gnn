"""
Certainly! Let me explain skipgrams in simple, everyday terms.

## What are Skipgrams?

Skipgrams are a way of looking at text that helps computers understand the relationships between words, even when they're not right next to each other.

### Basic Idea

Imagine you're reading a sentence and trying to understand the meaning of each word. You don't just look at the word immediately before or after it, right? You consider the context of the whole sentence. Skipgrams work on a similar principle.

### How it Works

1. **Skipping Words**: Instead of just looking at adjacent words, skipgrams "skip" over some words to capture relationships between words that are near each other, but not necessarily next to each other.

2. **Creating Pairs**: It creates pairs of words within a certain distance (called a "window") of each other.

3. **Learning Context**: This helps the computer learn that words appearing in similar contexts might have related meanings.

### Example

Let's take the sentence: "The quick brown fox jumps over the lazy dog"

With a window size of 2 (meaning we look at 2 words on either side), some skipgram pairs for "brown" would be:
- (brown, quick)
- (brown, fox)
- (brown, the)
- (brown, jumps)

Notice how it skipped over some words to create these pairs.

### Why It's Useful

1. **Better Understanding**: It helps computers grasp a more comprehensive understanding of language, similar to how humans understand context.

2. **Flexible Learning**: It can capture relationships between words even when they're not directly next to each other in a sentence.

3. **Improved AI Applications**: This technique is used in various AI applications like search engines, language translation, and recommendation systems.

In essence, skipgrams are like teaching a computer to read between the lines, helping it understand language in a more human-like way by considering the broader context of words in a sentence.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import re
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
    for i, word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                yield (word_to_idx[word], word_to_idx[words[j]])

window_size = 2
skip_grams = list(generate_training_data(words, window_size))
print(skip_grams)