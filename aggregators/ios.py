# coding: utf-8
###
 # @file   ios.py
 # @author Renaud Gaucher <renaud.gaucher@polytechnique.edu>
 #
 # @section DESCRIPTION
 #
 # Iterative Outlier scissors: from 
 # Byzantine-Resilient Decentralized Stochastic Optimization With Robust Aggregation Rules. 
 # Zhaoxian Wu, Tianyi Chen, and Qing Ling
###

from . import register
import tools
import torch
import numpy as np

# ---------------------------------------------------------------------------- #

def aggregate(gradients, f, honest_index, weights, **kwargs):
    """ IOS rule.
    Args:
        gradients       Non-empty list of gradients to aggregate
        f               Number of Byzantine gradients to tolerate
        honest_index    Index of the honest worker on which IOS is executed
        weights         weights associated with each of the gradients. Makes sense if they are positive
        ...       Ignored keyword-arguments
    Returns:
        Aggregated gradient using IOS
    """
    weights = torch.tensor(weights, device=gradients[0].device)
    trusted_set = weights>0
    gradients = torch.stack(gradients, dim=0)

    for _ in range(f):
        local_avg = (weights[trusted_set, None] * gradients[trusted_set,:]).sum(dim=0) / (weights[trusted_set].sum())

        differences = gradients - local_avg
        distances = differences.norm(dim=1)

        dist_max = 0
        index_argmax = 0
        for index, dist in enumerate(distances):
            if dist > dist_max and trusted_set[index] and index != honest_index:
                dist_max = dist
                index_argmax = index
        
        trusted_set[index_argmax] = False

    return (weights[trusted_set, None] * gradients[trusted_set,:]).sum(dim=0) / (weights[trusted_set].sum())


def check(gradients, honest_index, weights, communication_step=1, f=None, byz_weights=None, **kwargs):
    """ Check parameter validity for F-RG rules.
    Args:
        gradients           Non-empty list of gradients to aggregate
                f                   Number of Byzantine gradients to tolerate
                byz_weights         Weights of Byzantine gradients to tolerate (either f or byz_weights must be not None)
                honest_index        Index of the honest worker which proceed to the aggregation step
                weights             Weights used during the aggregation phase
                communication_step  Step-size used for the communication
        ...             Ignored keyword-arguments
    Returns:
            None if valid, otherwise error message string
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
    if not isinstance(honest_index, int):
        return f"Invalid honest index, got type = {type(honest_index)!r}, expected int"
    if isinstance(honest_index, int) and (honest_index < 0 or honest_index > (len(gradients)-1)):
        return f"Invalid honest index, got honest_index = {honest_index}, expected 0 ≤ honest_index ≤ {len(gradients) - 1}"
    if not (isinstance(weights, list) or isinstance(weights, torch.tensor) or isinstance(weights, np.array)):
        return f"Expected a list, a tensor or an array as weights, got {weights!r}"
    if not (len(weights)==len(gradients)):
        return f"Lenght of weights and gradients to aggregated do not match weights: {len(weights)} != {gradients.shape[0]} gradients"
    if (f is None and byz_weights is None):
        return f"Either f or byz_weights must be non None"
    if (f is not None and not isinstance(f,int)):
        return f"Type error for f {f!r}"
    
# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("IOS", aggregate, check)
