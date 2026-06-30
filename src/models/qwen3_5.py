import importlib
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import transformers.models.qwen3_5.modeling_qwen3_5 as modeling_qwen3_5
from lxt.efficient.patches import (
    dropout_forward,
    gated_mlp_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_attention,
    patch_method,
    rms_norm_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from src.models.base import BaseVLMWrapper


class Qwen3_5Wrapper(BaseVLMWrapper):
    def __init__(self, model: nn.Module, processor: Any):
        super().__init__(model, processor)
        self._get_original_attention_forward()
        self.is_mcvqa = True

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

        image_embeds = self.model.get_image_features(pixel_values,
                                                     image_grid_thw=image_grid_thw).pooler_output
        return image_embeds

    def merge_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        # image_embeds = image_outputs.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(text_embeds.device, text_embeds.dtype)
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_embeds
        )
        inputs_embeds = text_embeds.masked_scatter(image_mask, image_embeds)

        # image_embeds = torch.cat(image_embeds, dim=0).to(text_embeds.device, text_embeds.dtype)
        # image_mask, _ = self.model.model.get_placeholder_mask(
        #     input_ids, inputs_embeds=text_embeds, image_features=image_embeds
        # )
        # inputs_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def get_root_module(self) -> ModuleType:
        return importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")

    def get_patch_map(self) -> dict[Any, Any]:
        from src.explainers.lxt_utils.patches import get_smart_attention_patch

        attnLRP = {
            nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
            # modeling_qwen3_5.Qwen3_5VisionMLP: partial(patch_method, gated_mlp_forward),
            modeling_qwen3_5.Qwen3_5MLP: partial(patch_method, gated_mlp_forward),
            # modeling_qwen3_5.Qwen3_5RMSNorm: partial(patch_method, rms_norm_forward),
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            # modeling_qwen3_5.Qwen3_5RMSNormGated: partial(patch_method, rms_norm_forward),
            nn.Dropout: partial(patch_method, dropout_forward),
            modeling_qwen3_5: get_smart_attention_patch(),
        }


        return attnLRP


    def get_inputs(self, image,
                   text: str,
                   safe_pixels: int = 313600
                   ) -> dict[str, Any]:
        
        # 1. Base user message structure
        user_content = [
            {"type": "image", "image": image},
        ]

        # 2. Check if we are running MCVQA
        if self.is_mcvqa:
            # Define the strict system instruction
            system_instruction = (
                "You are a precise multiple-choice question answering assistant. "
                "Analyze the image and question. Output ONLY the single letter corresponding "
                "to the correct answer (e.g., A, B, C, or D). Do NOT include any introduction, "
                "explanation, punctuation, or extra whitespace. Just output the letter."
            )
            
            # Append the "Answer:" trigger to the text
            forced_user_text = f"{text}\nAnswer:"
            user_content.append({"type": "text", "text": forced_user_text})
            
            # Build messages WITH the system role
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ]
            
        else:
            # Standard behavior: just use the raw text and no system prompt
            user_content.append({"type": "text", "text": text})
            
            # Build messages WITHOUT the system role
            messages = [
                {"role": "user", "content": user_content}
            ]

        # 3. Apply the chat template and process
        templated_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.processor(
            text=[templated_text],
            images=[image],
            padding=True,
            max_pixels=safe_pixels,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        
        return inputs

    def get_tam_config(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tok = self.processor.tokenizer

        # 1. Dynamically fetch the shifted special tokens
        # This automatically grabs 248053, 248054, etc., for Qwen 3.5
        vision_start = tok.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tok.convert_tokens_to_ids("<|vision_end|>")
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")

        # 2. Core text tokens (These remain unchanged in the BPE vocab)
        newline = 198       # \n
        assistant = 77091   # assistant

        special_ids = {
            # Image is strictly bounded by <|vision_start|> and <|vision_end|>
            "img_id": [vision_start, vision_end],
            
            # Prompt starts after the image and ends at the assistant header: 
            # <|im_end|>\n<|im_start|>assistant
            "prompt_id": [vision_end, [im_end, newline, im_start, assistant]],
            
            # Answer starts right after the newline following the assistant header
            "answer_id": [[newline, im_start, assistant, newline], -1],
        }

        # 3. Dynamically calculate the Vision Grid (DO NOT hardcode to 24x24)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("TAM requires image_grid_thw for Qwen models")

        # Qwen Vision Transformer applies a 2x2 spatial downsampling.
        vision_shape = (
            image_grid_thw[0, 1].item() // 2, 
            image_grid_thw[0, 2].item() // 2
        )

        return {
            "vision_shape": vision_shape,
            "special_ids": special_ids,
        }
    
    def get_captum_forward(
        self,
        text_embeds,
        pixel_values,
        kwargs_dict,
        return_full_sequence: bool = False,
    ):  # <--- This catches attention_mask, image_grid_thw, etc.

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

    def _get_original_attention_forward(self):
        self._vision_attention_forward = modeling_qwen3_5.Qwen3_5VisionAttention.forward
        self._text_attention_forward = None

    def apply_patch(self):
        modeling_qwen3_5.Qwen3_5VisionAttention.forward = patch_vision_forward

    def remove_patch(self):
        modeling_qwen3_5.Qwen3_5VisionAttention.forward = self._vision_attention_forward

    @property
    def vision_module_name(self) -> str:
        return "Qwen3_5VisionAttention"

    @property
    def llm_module_name(self) -> str:
        return "Qwen3_5Attention"

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
    
    # Updated to use modeling_qwen3_5
    query_states, key_states = modeling_qwen3_5.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    # Use get_interface as defined in the newer Transformers library
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_qwen3_5.eager_attention_forward
    )

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
            output_attentions=True,  # Ensure attention weights are returned
            **kwargs,
        )
    else:
        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

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
                        # output_attentions=True,  # Ensure attention weights are returned
                        **kwargs,
                    )
                    for q, k, v in zip(*splits, strict=False)
                ),
                strict=False,
            ),
        )
        attn_output = torch.cat(attn_outputs, dim=1)

    # Save the visual attention weights
    self.saved_attn_weights = attn_weights

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output

