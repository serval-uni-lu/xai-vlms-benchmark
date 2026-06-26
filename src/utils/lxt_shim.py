
import torch
import transformers.pytorch_utils

# =======================================================================
# LXT DEPENDENCY SHIM: Inject missing HF functions (v4.57+ compatibility)
# =======================================================================
# (We do NOT mock BERT here. Because we fix the root cause below, 
# transformers.models.bert will import successfully on its own!)

# 1. Recreate find_pruneable_heads_and_indices
def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

# 2. Recreate prune_linear_layer
def prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

# 3. Recreate apply_chunking_to_forward
def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    return forward_fn(*input_tensors)

# --- THE MAGIC STEP ---
# Inject our recreated functions directly into the transformers module in memory
transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
transformers.pytorch_utils.prune_linear_layer = prune_linear_layer
transformers.pytorch_utils.apply_chunking_to_forward = apply_chunking_to_forward

