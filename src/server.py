import numpy as np
import torch
from utils import topk_eig, ModelQuantizer

def aggregate_subspace(scatter_mats, counts, k):
    """
    Aggregate scatter from clients and 
    compute global subspace."""
    total = sum(counts)
    S = sum((n/total) * Sm for Sm, n in zip(scatter_mats, counts))
    U, eigs = topk_eig(S, k)
    return U

def Fed_avg_aggregate_classifier(local_states, client_sizes,bits):
    """
    FedAvg on classifier 
    state dicts.
    if bits is set then we quantize the model
    """
    new_state = {}
    total = sum(client_sizes)

    if bits!=None:
        #print(f'load all quantized states for {bits} bits')
        # Initialize the quantizer to consider the quantized states
        quantizer = ModelQuantizer(num_bits=bits)
        # dequantize each quantized client states
        dequantized_local_states=[]
        for local_state in local_states:
            dequantized_local_states.append(quantizer.dequantize(local_state))
        local_states= dequantized_local_states

    for key in local_states[0].keys():
        avg = sum((n/total) * st[key].cpu().numpy() for st, n in zip(local_states, client_sizes))
        new_state[key] = torch.tensor(avg)
    return new_state

####################### This is for the top-k gradient selection #####################

def Fed_avg_top_k_gradient(global_w, deltas, client_sizes):
    total = sum(client_sizes)
    new_w = {}
    for name in global_w:
        # weighted sum of sparse deltas
        weighted = sum(n_c/total * d[name] for d,n_c in zip(deltas,client_sizes))
        new_w[name] = global_w[name] + weighted
    return new_w



