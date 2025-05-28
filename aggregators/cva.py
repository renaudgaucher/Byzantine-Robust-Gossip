

import tools
from . import register
import math, torch, aggregators

# ---------------------------------------------------------------------------- #

## NB : a priori l'implémentation sparse est OK (enfin pas pire que l'autre), mais a besoin de weights donc 
#pour test on comment

def aggregate(gradients, f, pivot=None, weights=None, honest_index=None, **kwargs):
  """ CVA rule.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    pivot           Pivot used for CVA. It is a string for an aggregation rule (PS)
    honest_index    Index of the honest worker on which CVA is executed (P2P)
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient using CVA
  """
  if honest_index is not None:
    #JS: P2P setting
    pivot_gradient = gradients[honest_index]
  else:
    #JS: PS setting, i.e., pivot is not None
    pivot_gradient = aggregators.gars.get(pivot).checked(gradients=gradients, f=f, **kwargs)
  
  if weights is None:  
    weights = torch.ones(len(gradients), device=gradients[0].device) / (len(gradients)-f)
  else:
    if len(weights) < len(gradients):
       weights += [weights[0]]*(len(gradients)-len(weights))
    weights = torch.tensor(weights, device=gradients[0].device)

  gradients = torch.stack(gradients, dim=0)
  differences = gradients - pivot_gradient
  distances = differences.norm(dim=1)
  trimming_trehshold = torch.topk(distances, f).values[-1]
  trimmed_differences = torch.where(
    distances[:, None] >= trimming_trehshold,  # Compare each norm to the threshold
    differences * 0,  #  set the largest differences to zero
    differences  # Otherwise, keep it unchanged
  )
  return pivot_gradient + (weights[:, None] *(trimmed_differences)).sum(dim=0)


def check(gradients, f, pivot=None, honest_index=None, **kwargs):
  """ Check parameter validity for CVA rule.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    pivot           Pivot used for CVA. It is a string for an aggregation rule (PS)
    honest_index    Index of the honest worker on which CVA is executed (P2P)
    ...             Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(honest_index, int) and honest_index is not None:
    return f"Invalid honest index, got type = {type(honest_index)!r}, expected int or None"
  if isinstance(honest_index, int) and (honest_index < 0 or honest_index > (len(gradients)-f-1)):
    return f"Invalid honest index, got honest_index = {honest_index!r}, expected 0 ≤ honest_index ≤ {len(gradients) - 1}"
  if not isinstance(pivot, int) and not isinstance(pivot, str) and pivot is not None:
    return f"Invalid honest index, got type = {type(pivot)!r}, expected int or str or None"
  if isinstance(pivot, int) and (pivot < 0 or pivot >= len(gradients)):
    return f"Invalid pivot, got pivot = {pivot!r}, expected 0 ≤ pivot ≤ {len(gradients)-1}"
# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (Pytorch version)
register("cva", aggregate, check)
