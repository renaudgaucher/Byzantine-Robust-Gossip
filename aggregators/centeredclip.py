

import tools
from . import register
import math, torch, aggregators

# ---------------------------------------------------------------------------- #
def aggregate(gradients, f, clip_thresh="adaptive", honest_index=None, weights=None, **kwargs):
  """ CG rule.
  Args:
    gradients       Non-empty list of gradients to aggregate
    f               Number of Byzantine gradients to tolerate
    honest_index    Index of the honest worker on which CG is executed (P2P only)
    weights         weights associated with each of the gradients. Makes sense if they are positive. If None, averaging is done
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient using CG
  """
  if weights is None:  
    weights = torch.ones(len(gradients), device=gradients[0].device) / (len(gradients)-f)
  else:
    if len(weights) < len(gradients):
       weights += [weights[0]]*(len(gradients)-len(weights))
    weights = torch.tensor(weights, device=gradients[0].device)

  if honest_index is not None:
    #JS: P2P setting
    pivot_gradient = gradients[honest_index]

  gradients = torch.stack(gradients, dim=0)
  differences = gradients - pivot_gradient
  distances = differences.norm(dim=1)

  if clip_thresh == "adaptive":
    clip_thresh = torch.sqrt(
      torch.mean(
        torch.topk(distances, k=distances.shape[0]-f, largest=False).values **2
        )
      )
  
  clipped_differences = torch.where(
    distances[:, None] > clip_thresh,  # Compare each norm to the threshold
    differences * (clip_thresh / distances[:, None]),  # Scale down the vector
    differences  # Otherwise, keep it unchanged
  )
  return (weights[:, None] * clipped_differences).sum(dim=0) + pivot_gradient


def check(gradients, f, clip_thresh="adaptive", honest_index=None, weights=None, **kwargs):
  """ Check parameter validity for CGplus rule.
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
  if isinstance(honest_index, int) and (honest_index < 0 or honest_index > (len(gradients)-1)):
    return f"Invalid honest index, got honest_index = {honest_index!r}, expected 0 ≤ honest_index ≤ {len(gradients) - 1}"
  if not isinstance(clip_thresh, int) and not isinstance(clip_thresh, float) and clip_thresh != "adaptive":
    return f"Invalid type for clipping threshold, got type =  {type(clip_thresh)}, expected int/float/ or str='adaptive'."
  if isinstance(clip_thresh, int) or isinstance(clip_thresh, float) and clip_thresh < 0:
    return f"Invalid value for clipping threshold, got {clip_thresh}, expected 0 ≤ clip_thresh."


def influence(honests, attacks, f, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    ...     Ignored keyword-arguments
  """
  return len(attacks) / (len(honests) + len(attacks))
# GAR registering

# Register aggregation rule (Pytorch version)
register("centeredclip", aggregate, check, influence=influence)
