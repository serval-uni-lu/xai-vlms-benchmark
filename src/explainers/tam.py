import numpy as np
import torch

from src.explainers import BaseExplainer
from src.explainers.tam_utils.tam import TAM
from src.models import BaseVLMWrapper
from src.utils.xai_utils import align_llm_visuals_to_pixels


class TAMExplainer(BaseExplainer):
    def __init__(self, model_wrapper: BaseVLMWrapper):
        super().__init__(model_wrapper)

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(
                inputs,
                return_logits=True,
            )

        logits = pred_results["logits"]
        generated_ids = pred_results["full_ids"].cpu()

        # --- DYNAMIC INDICES RESOLUTION ---
        seq_len_generated = len(logits)

        if target_indices is None:
            indices_to_compute = list(range(seq_len_generated))
        elif isinstance(target_indices, int):
            indices_to_compute = [target_indices]
        else:
            indices_to_compute = target_indices

        # Safety check: prevent out-of-bounds crashes
        indices_to_compute = [idx for idx in indices_to_compute if idx < seq_len_generated]

        tam_config = self.wrapper.get_tam_config(inputs)
        vis_inputs = np.array(image.convert("RGB"))

        # Get attributions
        img_scores_list = []
        text_attributions = []
        img_attributions = []

        # --- ITERATE ONLY OVER REQUESTED INDICES ---
        for idx in indices_to_compute:
            result = TAM(
                tokens=generated_ids.cpu().tolist(),
                vision_shape=tam_config["vision_shape"],
                logit_list=logits,
                special_ids=tam_config["special_ids"],
                vision_input=vis_inputs,
                processor=self.wrapper.processor,
                save_fn="",
                target_token=idx,  # Ask TAM to only explain this specific token
                img_scores_list=img_scores_list,
                eval_only=True,
                return_components=True,
            )

            img_attribution = result["img_map_norm"]
            text_attribution = result["txt_scores_raw"]

            text_attributions.append(text_attribution)
            img_attributions.append(img_attribution)

        # Convert lists to PyTorch tensors
        img_attributions = [torch.from_numpy(_).float() for _ in img_attributions]
        text_attributions = [torch.from_numpy(_).float() for _ in text_attributions]

        # Stack into shape: (num_targets, ...)
        img_attributions = torch.stack(img_attributions)

        # Flatten the spatial dimensions for the alignment function
        img_attributions = img_attributions.reshape(img_attributions.shape[0], -1)
        text_attributions = torch.stack(text_attributions)

        # Geometrically align the raw patches/tiles back to the expected pixel space
        img_attributions = align_llm_visuals_to_pixels(img_attributions, inputs, config=self.wrapper.model.config)

        return text_attributions, img_attributions
