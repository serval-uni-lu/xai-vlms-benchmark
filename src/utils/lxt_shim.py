import sys
import torch
import transformers.pytorch_utils

# =======================================================================
# LXT DEPENDENCY SHIM: Inject missing HF functions (v4.57+ compatibility)
# =======================================================================
def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

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

def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    return forward_fn(*input_tensors)

transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
transformers.pytorch_utils.prune_linear_layer = prune_linear_layer
transformers.pytorch_utils.apply_chunking_to_forward = apply_chunking_to_forward

import lxt.efficient  # Import LXT to intercept its patcher


# =======================================================================
# LXT DICTIONARY SHIM: Intercept and repair ALL_ATTENTION_FUNCTIONS
# =======================================================================
original_monkey_patch = lxt.efficient.monkey_patch

def safe_monkey_patch(module, patch_map, verbose=False):
    # 1. Run the original LXT patcher (which breaks the attention registry)
    original_monkey_patch(module=module, patch_map=patch_map, verbose=verbose)
    
    # 2. Immediately repair the damage across all loaded transformers models
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("transformers.models.") or mod_name.startswith("transformers.modeling_"):
            if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
                registry = getattr(mod, "ALL_ATTENTION_FUNCTIONS")
                
                # If LXT replaced it with a raw dict, upgrade it to support .get_interface()
                if type(registry) is dict:  
                    class LXTCompatRegistry(dict):
                        def get_interface(self, attn_implementation, eager_attention_forward):
                            return self.get(attn_implementation, eager_attention_forward)
                    
                    setattr(mod, "ALL_ATTENTION_FUNCTIONS", LXTCompatRegistry(registry))

# Overwrite LXT's function with our safe wrapper
lxt.efficient.monkey_patch = safe_monkey_patch
