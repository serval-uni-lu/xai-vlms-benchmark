import transformers
from lxt.efficient.patches import check_already_patched, patch_attention, wrap_attention_forward
from packaging import version


def safe_patch_attention(module):
    """
    A v5.10+ safe version of LXT's patch_attention.
    Instead of replacing ALL_ATTENTION_FUNCTIONS with a standard dict,
    it updates the existing AttentionInterface object in-place.
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    else:
        module.eager_attention_forward = new_forward
    
    # --- THE FIX ---
    for key, value in list(module.ALL_ATTENTION_FUNCTIONS.items()):
        new_forward = wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        else:
            module.ALL_ATTENTION_FUNCTIONS[key] = new_forward
            
    return True

def get_smart_attention_patch():
    """
    Dynamically returns the correct patch function based on the Transformers version.
    """
    if version.parse(transformers.__version__) >= version.parse("5.10.0"):
        return safe_patch_attention
    return patch_attention
