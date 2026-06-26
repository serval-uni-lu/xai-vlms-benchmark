from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    LlavaForConditionalGeneration,
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)

# Define a Union of all supported VLM classes
# This tells VS Code "The model is strictly one of these
# not just a generic Module"
#SupportedVLM = LlavaForConditionalGeneration | Qwen2_5_VLForConditionalGeneration | InternVLForConditionalGeneration
SupportedVLM = LlavaForConditionalGeneration | Qwen2_5_VLForConditionalGeneration | PreTrainedModel | Qwen3_5ForConditionalGeneration


# 2. Updated Class Definition (Python 3.12+)
# We use [ModelT: SupportedVLM] directly in the class name
class BaseVLMWrapper[ModelT: SupportedVLM](nn.Module, ABC):
    def __init__(self, model: ModelT, processor: Any):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model.eval()
        self.model_config = {}


    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to_device(self, device: str | torch.device) -> "BaseVLMWrapper":
        """
        Move the wrapper (and thus the HF model) to the given device.

        Returns self to allow chaining:
            wrapper.to_device("cuda")
        """
        device = torch.device(device)
        # nn.Module.to(...) will move all registered submodules, including hf_model
        super().to(device)
        return self

    @property
    @abstractmethod
    def vision_module_name(self) -> str:
        pass

    @property
    @abstractmethod
    def llm_module_name(self) -> str:
        pass

    @abstractmethod
    def _get_original_attention_forward(self):
        pass

    @abstractmethod
    def embed_images(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Projects raw pixel values into the LLM's embedding space.
        Returns: (Batch, Num_Image_Tokens, Hidden_Dim)
        """
        pass

    @abstractmethod
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return the nn.Module that corresponds to the text embedding layer
        of the underlying HF model.

        Must be implemented in subclasses, for example:
            return self.hf_model.get_input_embeddings()
        or
            return self.hf_model.vilt.text_embeddings
        depending on the architecture.
        """
        pass

    @abstractmethod
    def merge_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combines text and image embeddings into the final `inputs_embeds`
        sequence expected by the LLM.

        Returns:
            inputs_embeds: The merged tensor (Batch, Total_Len, Hidden)
        """
        pass

    @abstractmethod
    def get_inputs(self, image, text) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_root_module(self) -> ModuleType:
        pass

    @abstractmethod
    def get_patch_map(self) -> dict[Any, Any]:
        pass

    @abstractmethod
    def get_tam_config(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Returns model-specific configuration required by Transition Attention Maps (TAM).

        Expected keys:
            - vision_shape: Tuple[int, int] (H, W of the feature map)
            - special_ids: Dict[str, List[int]] (Token IDs for parsing structure)
            - vis_inputs: Any (The image representation TAM expects, e.g. numpy array)
        """
        pass

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # image_grid_thw: Optional[torch.Tensor] = None,
        # text_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: torch.Tensor | None = None,
        return_probs: bool = False,
        return_full_sequence: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        A unified forward pass that handles both standard inference
        and XAI injection.

        Logic Flow:
        1. If `inputs_embeds` is provided -> Use it directly (Fastest, for Merged XAI).
        2. Else if `text_embeds` is provided -> Combine with `pixel_values` (For Pixel XAI).
        3. Else -> Compute everything from scratch (Standard).
        """

        # Note: We use the base generic model call
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            # text_embeds=text_embeds,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            **kwargs,
        )

        logits = outputs.logits  # (B, seq_len, vocab_size)

        if return_full_sequence:  # (B, seq_len, vocab_size)
            if return_probs:
                return torch.nn.functional.log_softmax(logits, dim=-1)
            return logits

        # Default: The logit of the last token
        target_logits = logits[:, -1]  # (B, vocab_size)

        if return_probs:
            return torch.nn.functional.log_softmax(target_logits, dim=-1)

        return target_logits

    def get_captum_forward(
        self,
        text_embeds,
        pixel_values,
        kwargs_dict,
        return_full_sequence: bool = False,
    ):  # <--- This catches attention_mask, image_grid_thw, etc.

        # We capture 'kwargs' here.
        # Python closures automatically allow the inner function to access 'kwargs'

        if pixel_values is not None:
            image_embeds = self.embed_images(pixel_values, **kwargs_dict)
        else:
            raise ValueError("Need pixel_values tensor")

        input_ids = kwargs_dict.get("input_ids", None)
        if input_ids is None:
            raise ValueError("input_ids tensor is required")

        # text_embeds = self.embed_text(input_ids)
        inputs_embeds = self.merge_embeddings(text_embeds, image_embeds, input_ids)

        return self.forward(
            inputs_embeds=inputs_embeds,
            return_probs=True,
            return_full_sequence=return_full_sequence,
            **kwargs_dict,
        )

    def predict(
        self,
        inputs: dict[str, Any],
        # input_ids: Optional[torch.Tensor] = None,
        # pixel_values: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        return_logits: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        #assert hasattr(self.model, "generate"), "Model must support .generate()"

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=return_logits,
                use_cache=use_cache,
                **kwargs,
            )

        # 3. Decode

        generated_ids = outputs.sequences[0]  # (Seq_Len,)
        # Filter out the input prompt to get only the new tokens
        prompt_len = inputs["input_ids"].shape[1]
        # Note: In some HF models, generate returns full sequence; others just new tokens.
        # You need a safety check here depending on model family.
        new_ids = generated_ids[prompt_len:] if generated_ids.shape[0] > prompt_len else generated_ids

        decoded_text = self.processor.batch_decode(
            new_ids.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if return_logits:
            logits = [self.model.lm_head(feats[-1]) for feats in outputs.hidden_states]
            return {
                "text": decoded_text,
                "full_ids": generated_ids.cpu(),  # (Seq_Len,)
                "new_ids": new_ids.cpu(),  # (Ans_Len,)
                "logits": logits,
            }

        return {"text": decoded_text, "full_ids": generated_ids, "new_ids": new_ids}

    def get_forward_fn(
        self,
        inputs,
        mode: Literal["image", "text", "joint"] = "joint",
        **kwargs,
    ):
        """
        Creates a specialized forward function compatible with Captum/Perturbation Metrics.

        The returned function strictly matches the signature expected by Captum:
        - mode="image" -> f(pixel_values)
        - mode="text"  -> f(text_embeds)
        - mode="joint" -> f(pixel_values, text_embeds)
        """
        device = self.device
        inputs = inputs.to(device)
        # image_grid_thw = inputs.get("image_grid_thw", None)

        # If we are explaining Image only, Text is static. Compute embeddings once.
        if mode == "image":
            input_ids = inputs.get("input_ids", None)
            attention_mask = inputs.get("attention_mask", None)
            text_embeds = self.embed_text(input_ids)
            if text_embeds is None:
                raise ValueError("Need input_ids to build text embeddings")

            def forward_pixels(pixel_values):
                return self.forward(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    text_embeds=text_embeds,
                    attention_mask=attention_mask,
                    return_probs=False,
                    **kwargs,
                )

            return forward_pixels

        elif mode == "text":
            pixel_values = inputs.get("pixel_values", None)
            if pixel_values is None:
                raise ValueError("Need pixel_values to build text embeddings")

            def forward_input_ids(input_ids, attention_mask):
                text_embeds = self.embed_text(input_ids)

                return self.forward(
                    input_ids=None,
                    text_embeds=text_embeds,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_probs=False,
                    **kwargs,
                )

            return forward_input_ids

        elif mode == "joint":
            # --- Case C: Explaining Both (Input: (Pixels, Text_Embeds)) ---
            def forward_joint(input_ids, pixel_values, attention_mask):
                text_embeds = self.embed_text(input_ids)
                return self.forward(
                    input_ids=None,
                    pixel_values=pixel_values,
                    text_embeds=text_embeds,
                    attention_mask=attention_mask,
                    return_probs=False,
                    **kwargs,
                )

            return forward_joint

        else:
            raise ValueError(f"Unknown mode {mode}")

    @abstractmethod
    def apply_patch(self):
        pass

    @abstractmethod
    def remove_patch(self):
        pass

    @property
    @abstractmethod
    def special_token_ids(self) -> list[Any]:
        pass
