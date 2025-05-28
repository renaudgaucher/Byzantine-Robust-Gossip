from . import register
import tools
import torch
import numpy as np

# ---------------------------------------------------------------------------- #
def make_defense(robust_summand):
    # Make an aggregation routine that follows the RG framework using the robust summand taken as input
    
    def aggregate(gradients, honest_index, weights, communication_step=1, f=None, byz_weights=None, **kwargs):
        """ robust_summand-RG aggregation step on honest node 'honest_index'

        Args:
            gradients           Non-empty list of gradients to aggregate
            f                   Number of Byzantine gradients to tolerate
            byz_weights         Weights of Byzantine gradients to tolerate (either f or byz_weights must be not None)
            honest_index        Index of the honest worker which proceed to the aggregation step
            weights             Weights used during the aggregation phase
            communication_step  Step-size used for the communication
            ...       Ignored keyword-arguments
        Returns:
            Aggregated gradient using robust_summand-RG
        """
        weights=torch.tensor(weights, device=gradients[0].device)

        if byz_weights is None:  
            byz_weights = weights[-1]*f # messages due to byzantine are at the end (see P2PSparse.py)

        pivot_gradient = gradients[honest_index]
        gradients = torch.stack(gradients, dim=0)
        differences = pivot_gradient - gradients

        robust_aggregate = robust_summand(
            weights=weights, gradients=differences, byz_weights=byz_weights, honest_index=honest_index
            )
        
        return pivot_gradient - communication_step * robust_aggregate
    
    # return the constructed F-RG aggregation function
    return aggregate



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
    if  (byz_weights is not None) and (byz_weights < 0 or byz_weights > (sum(weights))):
        return f"Invalid byz_weights, got byz_weights = {byz_weights}, expected 0 ≤ byz_weights ≤ {sum(weights)}"
  
  
# ---------------------------------------------------------------------------- #
# GAR registering


### CSplus from our article

def cs_plus(weights, gradients, byz_weights, **kwargs):

    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]


    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - 2*byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # nb: weights[idx:] >= byz_weights
    
    # Compute the clipping threshold
    if idx < 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf

    elif cumulative_weights[idx] == target_weight:
        clipping_threshold = sorted_distances[idx]
    elif cumulative_weights[idx] > target_weight:
        if idx-1 >= 0:
            clipping_threshold = sorted_distances[idx-1]
        else:
            clipping_threshold =  0
    else:
        raise ValueError("Unexpected behavior in computing the adaptive clipping threshold")
    
    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    clipped_differences = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_differences).sum(dim=0)
    
### GTS, leads to NNA

def gts(weights, gradients, byz_weights, **kwargs):

    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # weights(idx:) >= byz_weights
    
    rest_weight = 0
    if cumulative_weights[idx] > target_weight:
        rest_weight =  cumulative_weights[idx] - target_weight
    
    sorted_gradients = gradients[sorted_indices,:]
    return (sorted_weights[:idx, None] * sorted_gradients[:idx,:]).sum(dim=0) + sorted_gradients[idx,:] * rest_weight


### CShe practical from [He, Karimirredy and Jaggi 2022]

def cs_he(weights, gradients, byz_weights, **kwargs):

    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # weights(idx:) >= byz_weights
    
    # Compute the adaptive clipping threshold of He et al. 
    if idx == 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf
    else:
        clipping_threshold = ((sorted_weights[:idx] * sorted_distances[:idx]**2).sum(dim=0) / byz_weights).sqrt().item()

    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    clipped_gradients = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_gradients).sum(dim=0)


#### Plain non robust averaging

def average(weights, gradients, **kwargs):
    return (weights[:, None] * gradients).sum(dim=0)


# Register aggregation rule
for name, func in (("CSplus_RG", cs_plus), ("GTS_RG", gts), ("CShe_RG", cs_he), ("average_sparse", average)):
  register(name, make_defense(func), check)