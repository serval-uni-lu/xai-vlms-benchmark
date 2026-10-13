from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.models.base import BaseVLMWrapper  # Assuming previous code is here


@dataclass
class XAIResult:
    """
    A unified container for attribution results.
    Keeps data on CPU to save GPU memory during analysis.
    """

    # The raw attribution scores
    image_attributions: torch.Tensor  # Shape: (C, H, W) or (Num_Patches,)
    text_attributions: torch.Tensor  # Shape: (Seq_Len,)

    # Metadata for visualization
    input_tokens: list[str]  # The actual text tokens
    pixel_values: torch.Tensor  # The original image (for plotting overlay)

    # Validation data
    score: float  # The total attribution score (sum)
    target_token: str  # What token were we explaining?

    def cpu(self):
        """Moves all tensors to CPU to avoid OOM during benchmarking."""
        self.image_attributions = self.image_attributions.detach().cpu()
        self.text_attributions = self.text_attributions.detach().cpu()
        if isinstance(self.pixel_values, torch.Tensor):
            self.pixel_values = self.pixel_values.detach().cpu()
        return self


class BaseExplainer(ABC):
    def __init__(self, model_wrapper: BaseVLMWrapper):
        self.wrapper = model_wrapper
        self.device = model_wrapper.device

    def attribute(
        self, image, text: str, target_indices: int | list[int] | None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The concrete public method. This guarantees post-processing (CPU transfer)
        no matter what the subclass does.
        """
        # 1. Call the subclass's unique logic
        text_attrs, img_attrs = self._attribute(image, text, target_indices, **kwargs)

        # 2. Enforce safety
        if isinstance(text_attrs, torch.Tensor):
            text_attrs = text_attrs.detach().cpu().float()
        if isinstance(img_attrs, torch.Tensor):
            img_attrs = img_attrs.detach().cpu().float()

        return text_attrs, img_attrs

    @abstractmethod
    def _attribute(
        self, image, text: str, target_indices: int | list[int] | None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CHILD CLASSES MUST IMPLEMENT THIS INSTEAD.
        Implement the specific XAI logic here.
        """
        pass
