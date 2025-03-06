import torch 
import numpy as np 

from transformer import define_gpt2_model

def pretty_print(dictionary, ident=""):
    for key, val in dictionary.items():
        if hasattr(val, "shape"):
            print()

        if isinstance(val, dict):
            pretty_print(val, ident + "  ")
        
        if isinstance(val, list):
            for i, item in enumerate(val):
                pretty_print({i: item}, ident + "  ")
    

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    pass 