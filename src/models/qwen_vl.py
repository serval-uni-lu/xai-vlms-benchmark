import importlib
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as modeling_qwen2_5_vl
from lxt.efficient.patches import (
    dropout_forward,
    gated_mlp_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_attention,
    patch_method,
    rms_norm_forward,
)
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from src.models.base import BaseVLMWrapper


class QwenVLWrapper(BaseVLMWrapper):
    def __init__(self, model: nn.Module, processor: Any):
        super().__init__(model, processor)
        self._get_original_attention_forward()

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def embed_images(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("QwenVl needs image_grid_thw tensor")

        if pixel_values.dim() == 3:
            batch_size = pixel_values.shape[0]

            # 1. Flatten pixel_values back to 2D for Qwen
            # (Batch, Patches, Dim) -> (Batch*Patches, Dim)
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])

            # 2. Expand image_grid_thw to match the batch size
            # Current grid shape: (1, 3) -> We need (Batch_Size, 3)
            if image_grid_thw.shape[0] == 1 and batch_size > 1:
                image_grid_thw = image_grid_thw.repeat(batch_size, 1)

        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
        return image_embeds

    def merge_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        image_embeds = torch.cat(image_embeds, dim=0).to(text_embeds.device, text_embeds.dtype)
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_embeds
        )
        inputs_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def get_root_module(self) -> ModuleType:
        return importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")

    def get_patch_map(self) -> dict[Any, Any]:
        attnLRP = {
            nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
            modeling_qwen2_5_vl.Qwen2_5_VLMLP: partial(patch_method, gated_mlp_forward),
            modeling_qwen2_5_vl.Qwen2MLP: partial(patch_method, gated_mlp_forward),
            Qwen2RMSNorm: partial(patch_method, rms_norm_forward),
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            # modeling_qwen2_5_vl.Qwen2_5_VLRMSNorm: partial(patch_method, rms_norm_forward),
            nn.Dropout: partial(patch_method, dropout_forward),
            modeling_qwen2_5_vl: patch_attention,
        }
        return attnLRP

    def get_inputs(self, image, text,
                   safe_pixels=313600
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
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            max_pixels=safe_pixels,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        return inputs

    def get_tam_config(self, inputs: dict[str, Any]) -> dict[str, Any]:

        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            # Fallback if batching stripped it, or default size
            raise ValueError("TAM requires image_grid_thw for Qwen")

        vision_shape = (image_grid_thw[0, 1] // 2, image_grid_thw[0, 2] // 2)

        special_ids = {
            "img_id": [151652, 151653],
            "prompt_id": [151653, [151645, 198, 151644, 77091]],
            "answer_id": [[198, 151644, 77091, 198], -1],
        }

        return {
            "vision_shape": vision_shape,
            "special_ids": special_ids,
            # "vis_inputs": vis_inputs
        }

    def _get_original_attention_forward(self):
        self._vision_attention_forward = modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward
        self._text_attention_forward = None

    def apply_patch(self):
        modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = patch_vision_forward

    def remove_patch(self):
        modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = self._vision_attention_forward

    @property
    def vision_module_name(self) -> str:
        return "Qwen2_5_VLVisionAttention"

    @property
    def llm_module_name(self) -> str:
        return "Qwen2_5_VLAttention"

    @property
    def special_token_ids(self) -> list:
        tok = self.processor.tokenizer
        special_token_ids = [
            tok.convert_tokens_to_ids("<|image_pad|>"),
            tok.convert_tokens_to_ids("<|vision_start|>"),
            tok.convert_tokens_to_ids("<|vision_end|>"),
            tok.convert_tokens_to_ids("<|im_start|>"),
            tok.convert_tokens_to_ids("<|im_end|>"),
        ]
        return special_token_ids


def patch_vision_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = modeling_qwen2_5_vl.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # if modeling_qwen2_5_vl.is_flash_attention_requested(self.config):
    if self.config._attn_implementation == "flash_attention_2":
        # Flash Attention: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs, attn_weights = map(
            list,
            zip(
                *(
                    attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )
                    for q, k, v in zip(*splits, strict=False)
                ),
                strict=False,
            ),
        )
        attn_output = torch.cat(attn_outputs, dim=1)

    # Save here the visual attention weights
    self.saved_attn_weights = attn_weights
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


def load_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct", attn_implementation=None, gpu_node=0):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_node}",
        attn_implementation=attn_implementation,
    )
    return model, processor
