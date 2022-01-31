import numpy as np
import random
from random import shuffle

def read_dinos():
    data = open('0_dataset/Dinosaurs/dinos.txt', 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

    return char_to_ix, ix_to_char

if __name__ == "__main__":
    char_to_ix, ix_to_char = read_dinos()
    print(char_to_ix)