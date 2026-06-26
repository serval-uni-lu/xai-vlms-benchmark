import importlib
import math
from functools import partial
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import transformers.models.internvl.modeling_internvl as modeling_internvl
from lxt.efficient.models.qwen3 import attnLRP as qwen3_attnlrp
from lxt.efficient.patches import dropout_forward, patch_attention, patch_method, rms_norm_forward
from torch.nn import Dropout
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from src.models.base import BaseVLMWrapper


class InternVLWrapper(BaseVLMWrapper):
    def __init__(self, model: nn.Module, processor: Any):
        super().__init__(model, processor)
        # self._get_original_attention_forward()

    def get_inputs(self, image, text
                    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            return_dict=True,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        return inputs

    @property
    def vision_module_name(self) -> str:
        return "InternVLVisionAttention"

    @property
    def llm_module_name(self) -> str:
        return "Qwen3Attention"

    @property
    def special_token_ids(self) -> list:
        tok = self.processor.tokenizer
        special_token_ids = [
            tok.convert_tokens_to_ids("<IMG_CONTEXT>"),
            tok.convert_tokens_to_ids("<img>"),
            tok.convert_tokens_to_ids("</img>"),
            tok.convert_tokens_to_ids("<|im_start|>"),
            tok.convert_tokens_to_ids("<|im_end|>"),
        ]
        return special_token_ids

    def apply_patch(self):
        pass

    def embed_images(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:

        vision_feature_layer = kwargs.get("vision_feature_layer")
        vision_feature_select_strategy = kwargs.get("vision_feature_select_strategy")

        image_features = self.model.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            return_dict=True,
        )
        return image_features

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def merge_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        image_embeds = image_embeds.to(text_embeds.device, text_embeds.dtype)
        special_image_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_embeds
        )
        inputs_embeds = text_embeds.masked_scatter(special_image_mask, image_embeds)
        return inputs_embeds

    def get_captum_forward(
        self,
        text_embeds,
        pixel_values,
        kwargs_dict,
        return_full_sequence: bool = False,
    ):  # <--- This catches attention_mask, image_grid_thw, etc.

        # We capture 'kwargs' here.
        # Python closures automatically allow the inner function to access 'kwargs'

        local_kwargs = kwargs_dict.copy()

        if pixel_values is not None:
            image_embeds = self.embed_images(pixel_values, **local_kwargs)
        else:
            raise ValueError("Need pixel_values tensor")

        input_ids = local_kwargs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("input_ids tensor is required")

        # text_embeds = self.embed_text(input_ids)
        inputs_embeds = self.merge_embeddings(text_embeds, image_embeds, input_ids)

        local_kwargs["input_ids"] = None

        return self.forward(
            inputs_embeds=inputs_embeds,
            return_probs=True,
            return_full_sequence=return_full_sequence,
            **local_kwargs,
        )

    def get_patch_map(self) -> dict[str, Any]:
        attnLRP = {
            modeling_internvl.InternVLVisionRMSNorm: partial(patch_method, rms_norm_forward),
            # modeling_internvl.InternVLVisionMLP: partial(patch_method, gated_mlp_forward),
            Dropout: partial(patch_method, dropout_forward),
            modeling_internvl: patch_attention,
        }
        attnLRP.update(qwen3_attnlrp)
        return attnLRP

    def _get_original_attention_forward(self):
        return super()._get_original_attention_forward()

    def get_root_module(self) -> ModuleType:
        return importlib.import_module("transformers.models.internvl.modeling_internvl")

    def get_tam_config(self, inputs: dict[str, Any]) -> dict[str, Any]:

        # InternVL has different special ids, please vis inputs['input_ids'] for special ids

        special_ids = {
            "img_id": [151671],
            "prompt_id": [151653, [151645, 198, 151644, 77091]],
            "answer_id": [[198, 151644, 77091, 198], -1],
        }

        tiles_dim = getattr(self.model.config, "image_seq_length", 256)
        tile_size = int(math.sqrt(tiles_dim))
        vision_shape = (tile_size, tile_size)

        return {
            "vision_shape": vision_shape,
            "special_ids": special_ids,
            # "vis_inputs_shape": (448, 448)
        }

    def remove_patch(self):
        return super().remove_patch()


def load_model(model_id="OpenGVLab/InternVL3_5-2B-HF", attn_implementation=None, gpu_node=0):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_node}",
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).eval()
    return model, processor
