from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)

from src.models import BaseVLMWrapper, InternVLWrapper, LlavaWrapper, Qwen3_5Wrapper, QwenVLWrapper


class VLMType:
    QWEN2_5_VL = "qwen2_5_vl"
    QWEN3_5 = "qwen3_5"
    INTERNVL = "internvl"
    LLAVA = "llava"


def load_vlm(
    model_config: dict[str, Any],
    attn_implementation: str | None = None,
    gpu_node: int = 0,
    output_attentions: bool = False,
) -> BaseVLMWrapper:
    """
    Unified factory to load a VLM and wrap it based on a YAML configuration.
    """
    model_id = model_config["model_id"]
    vlm_type = model_config.get("vlm_type", model_config["name"])
    trust_remote = model_config.get("trust_remote_code", False)

    print(f"[*] Loading model architecture: {model_id} on GPU {gpu_node}...")

    # 1. Setup Quantization (Equivalent to your LoaderConfig)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Prepare Loading Arguments
    if gpu_node == "auto":
        device_map = "auto"
        num_gpus = torch.cuda.device_count()

        # Define your memory limits
        gpu_0_limit = "10GiB"       # Restricted memory for compute
        other_gpu_limit = "24GiB" # "24GiB"  # Max memory for the rest

        # Dynamically build the dictionary
        custom_max_memory = {
            i: gpu_0_limit if i == 0 else other_gpu_limit 
            for i in range(num_gpus)
        }
        custom_max_memory = None

    elif isinstance(gpu_node, int):
        device_map= f"cuda:{gpu_node}"
        custom_max_memory = None
    else:
        raise ValueError(f"This value for gpu node is not implemented: {gpu_node}")
    load_kwargs = {
        "device_map": device_map,
        "max_memory": custom_max_memory,
        "trust_remote_code": trust_remote,
        "dtype": torch.bfloat16,
        "quantization_config": bnb_config,
        "attn_implementation": attn_implementation,
    }

    # 3. Load Processor
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote)
        if VLMType.INTERNVL in vlm_type:
            # Reduce the number of tiles to avoid VRAM and GPU overcomsumption
            processor.image_processor.max_patches = 2

    except Exception as e:
        raise ValueError(f"Failed to load processor for {model_id}: {e}") from e

    # 4. Load Model
    try:
        if VLMType.QWEN2_5_VL in vlm_type:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        elif VLMType.QWEN3_5 in vlm_type:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        elif VLMType.INTERNVL in vlm_type:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        elif VLMType.LLAVA in vlm_type:
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        else:
            print(f"[!] Warning: Falling back to AutoModelForCausalLM for {vlm_type}")
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load model for {model_id}: {e}") from e

    model.eval()

    # # 5. Handle Attention Flags (Specifically for Rollout on LLaVA)
    # if output_attentions and vlm_type == VLMType.LLAVA:
    #     model.config.vision_config.output_attentions = True
    #     if hasattr(model.vision_tower, "config"):
    #         model.vision_tower.config.output_attentions = True
    #         model.vision_tower.vision_model.config.output_attentions = True

    # 5. Handle Attention Flags (Specifically for Rollout on LLaVA)
    if output_attentions and vlm_type == VLMType.LLAVA:
        # We use getattr/setattr to avoid static type checking errors on dynamic HF attributes
        model.config.output_attentions = True
        
        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is not None:
            # Set on the tower config
            if hasattr(vision_tower, "config"):
                vision_tower.config.output_attentions = True
            
            # Set on the inner vision_model config if it exists
            v_model = getattr(vision_tower, "vision_model", None)
            if v_model and hasattr(v_model, "config"):
                v_model.config.output_attentions = True

    # 6. Dispatch to Correct Wrapper
    if VLMType.QWEN2_5_VL in vlm_type:
        return QwenVLWrapper(model, processor)
    
    elif VLMType.QWEN3_5 in vlm_type:
        return Qwen3_5Wrapper(model, processor)

    elif VLMType.INTERNVL in vlm_type:
        return InternVLWrapper(model, processor)

    elif VLMType.LLAVA in vlm_type:
        return LlavaWrapper(model, processor)

    else:
        raise NotImplementedError(f"Wrapper for {vlm_type} is defined but not instantiated in factory.")
