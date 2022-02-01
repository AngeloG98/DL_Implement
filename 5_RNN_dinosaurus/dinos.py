import numpy as np
import random
from random import shuffle

def read_dinos(filename):
    #load vocab
    data = open(filename, 'r').read()
    data = data.lower()
    chars = list(set(data))
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    #load words
    with open(filename) as f:
        words = f.readlines()
    words = [x.lower().strip() for x in words]

    return char_to_ix, ix_to_char, words


    
