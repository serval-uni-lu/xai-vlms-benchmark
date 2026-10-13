import importlib
from functools import partial
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import transformers.models.clip.modeling_clip as modeling_clip
from lxt.efficient.models.llama import attnLRP as llama_attnlrp
from lxt.efficient.patches import layer_norm_forward, patch_attention, patch_method
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

from src.models.base import BaseVLMWrapper


class LlavaWrapper(BaseVLMWrapper):
    def __init__(self, model: nn.Module, processor: Any):
        super().__init__(model, processor)
        # self._get_original_attention_forward()

    def get_inputs(self, image, text,
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
        return "CLIPAttention"

    @property
    def llm_module_name(self) -> str:
        return "LlamaAttention"

    @property
    def special_token_ids(self) -> list:
        tok = self.processor.tokenizer
        special_token_ids = [
            tok.convert_tokens_to_ids("<image>"),
            tok.convert_tokens_to_ids("<s>"),
            # tok.convert_tokens_to_ids("<|vision_start|>"),
            # tok.convert_tokens_to_ids("<|vision_end|>"),
            # tok.convert_tokens_to_ids("<|im_start|>"),
            # tok.convert_tokens_to_ids("<|im_end|>")
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
        ).pooler_output
        return image_features

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def merge_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        image_embeds = torch.cat(image_embeds, dim=0).to(text_embeds.device, text_embeds.dtype)
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

    def get_patch_map(self) -> dict[Any, Any]:
        attnLRP = {
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            modeling_clip: patch_attention,
        }
        attnLRP.update(llama_attnlrp)
        return attnLRP

    def _get_original_attention_forward(self):
        return super()._get_original_attention_forward()

    def get_root_module(self) -> ModuleType:
        return importlib.import_module("transformers.models.llava.modeling_llava")

    def get_tam_config(self, inputs: dict[str, Any]) -> dict[str, Any]:

        special_ids = {
            "img_id": [[29901, 29871], [29871, 13]],
            "prompt_id": [[29871, 13], [319, 1799]],
            "answer_id": [[319, 1799, 9047, 13566, 29901], -1],
        }

        vision_shape = (24, 24)

        return {
            "vision_shape": vision_shape,
            "special_ids": special_ids,
        }

    def remove_patch(self):
        return super().remove_patch()


def load_model(
    model_id="llava-hf/llava-1.5-7b-hf",
    attn_implementation=None,
    gpu_node=0,
    output_attentions=False,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_node}",
        attn_implementation=attn_implementation,
        output_attentions=output_attentions,
        trust_remote_code=True,
    ).eval()

    if output_attentions:
        model.config.vision_config.output_attentions = True
        model.vision_tower.config.output_attentions = True
        model.vision_tower.vision_model.config.output_attentions = True

    return model, processor
