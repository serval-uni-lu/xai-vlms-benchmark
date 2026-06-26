import torch
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit
from zennit.composites import LayerMapComposite

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from src.models.factory import load_vlm


class LXTExplainer(BaseExplainer):
    def __init__(
        self,
        model_wrapper: BaseVLMWrapper,
        use_zennit=False,
        token_wise=True,
    ):
        # super().__init__(model_wrapper)
        self.wrapper = model_wrapper
        self.zennit_comp = None
        self.token_wise = token_wise

        # Check & Apply Patches (Idempotent)
        # This will update self.wrapper if a reload occurred
        self._ensure_patched(
            verbose=True,
            use_zennit=use_zennit,
        )

        # Update parent with the potentially new wrapper
        super().__init__(self.wrapper)

    def _ensure_patched(
        self,
        verbose: bool = False,
        use_zennit: bool = False,
    ):

        # Get Patch Info from the wrapper (Delegated to wrapper logic)
        if not hasattr(self.wrapper, "get_patch_map"):
            print("[!] Wrapper does not support LXT patching auto-detection.")
            return

        module_root = self.wrapper.get_root_module()
        patch_dict = self.wrapper.get_patch_map()
        model_id = self.wrapper.model.config.name_or_path
        # vlm_type = self.wrapper.model.config.model_type

        # --- IDEMPOTENCY CHECK ---
        # We check a flag on the module itself to see if we already touched it
        if getattr(module_root, "__lxt_is_patched__", False):
            # Already patched! Do not reload model.
            return

        print(f"[*] Applying LXT Monkey Patches to {module_root}...")
        monkey_patch(module=module_root, patch_map=patch_dict, verbose=verbose)

        if use_zennit:
            monkey_patch_zennit(verbose=verbose)

        # Mark as patched
        module_root.__lxt_is_patched__ = True

        try:
            # 3. Load Model (It will now instantiate using PATCHED classes)
            # We use your factory from before
            print(f"[*] Reloading patched model: {model_id}")
            device = self.wrapper.device

            # Retrieve the stored YAML config
            stored_config = getattr(self.wrapper, "model_config", None)
            if stored_config is None:
                raise RuntimeError("LXT requires model_config to be stored on the wrapper to reload.")

            del self.wrapper
            self.wrapper = load_vlm(
                **stored_config,
                # model_config=stored_config,
                # attn_implementation=None,
            )
            self.wrapper.model_config = stored_config
            self.wrapper.to_device(device)
            torch.cuda.empty_cache()

        finally:
            print("[!] Exiting Patch Context.")
            print("[!] WARNING: Python classes are still modified in memory.")
            print("[!] It is highly recommended to restart the kernel/process before running a different XAI method.")
            # Unlike hooks, monkey patching classes is very hard to undo robustly
            # without reloading the 'transformers' library entirely.

        # Configure LXT (Zennit etc.)
        # Assuming you have a helper for this
        zennit_comp = self._configure_lxt(
            self.wrapper.model,
            use_zennit=use_zennit,
        )
        self.zennit_comp = zennit_comp

    def _configure_lxt(self, model: torch.nn.Module, use_zennit=False):

        zennit_comp = None

        if use_zennit:
            # Define rules for the Conv2d and Linear layers using 'zennit'
            conv_gamma = 100
            lin_gamma = 0.05
            # LayerMapComposite maps specific layer types to specific LRP rule implementations
            zennit_comp = LayerMapComposite(
                [
                    (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
                    (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
                ]
            )

            # monkey_patch_zennit(verbose=True)

        # Set up the model for the explanation task
        model.train()  # Switch to train mode to enable  gradient flow
        model.gradient_checkpointing_enable()  # Optional: saves memory

        # Deactivate gradients on model parameters to save memory and ensure LRP rules apply
        for param in model.parameters():
            param.requires_grad = False

        if zennit_comp is not None:
            # Register the composite rules with the model
            zennit_comp.register(model)
        return zennit_comp

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 1. SETUP PRISTINE INPUTS
        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(
                inputs=inputs,
                return_logits=False,
                **kwargs,
            )
        full_ids = pred_results["full_ids"].to(self.device)

        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]
        gen_len = t_end - t_start

        # --- DYNAMIC INDICES RESOLUTION ---
        if target_indices is None:
            indices_to_compute = list(range(gen_len))
        elif isinstance(target_indices, int):
            indices_to_compute = [target_indices]
        else:
            indices_to_compute = target_indices

        # Safety check
        indices_to_compute = [idx for idx in indices_to_compute if idx < gen_len]

        # Build full sequence inputs
        inputs["input_ids"] = full_ids.clone()
        if inputs["input_ids"].ndim == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        kwargs_dict = {k: v for k, v in inputs.items() if k not in ["pixel_values"]}

        # Pre-embed text so we don't have to do it in the loop
        base_text_embeds = self.wrapper.embed_text(inputs["input_ids"]).clone().detach()
        base_pixel_values = inputs["pixel_values"].clone().detach()

        token_attributions = []
        pixel_attributions = []

        model_type = self.wrapper.model.config.model_type.lower()

        # We need the exact token IDs generated to gather the correct logit
        generated_token_ids = full_ids[t_start:t_end]

        # 2. BACKWARD PASS ENGINE (Gradient * Activation)
        if not self.token_wise:
            # --- SENTENCE-LEVEL (AGGREGATED) ---
            # Run one forward pass
            pixel_values = base_pixel_values.clone().requires_grad_(True)
            text_embeds = base_text_embeds.clone().requires_grad_(True)

            self.wrapper.model.zero_grad()

            log_logits = self.wrapper.get_captum_forward(
                text_embeds=text_embeds,
                pixel_values=pixel_values,
                kwargs_dict=kwargs_dict,
                return_full_sequence=True,
            )

            # Causal Shift: Grab the logits that predicted the generated tokens
            gen_logits = log_logits[:, t_start - 1 : t_end - 1, :]

            # Gather target probabilities
            gathered_logits = gen_logits[0, torch.arange(gen_len), generated_token_ids]

            # Sum only the requested indices
            sentence_score = gathered_logits[indices_to_compute].sum()
            sentence_score.backward(retain_graph=False)  # False is safe here!

            # LRP Rule: Relevance = Gradient * Activation
            if "qwen" in model_type:
                rel_img = (pixel_values.grad * pixel_values).float().sum(-1).detach().cpu()
            elif "internvl" in model_type:
                rel_img = (pixel_values.grad * pixel_values).float().sum(-3).detach().cpu()
            elif "llava" in model_type:
                rel_img = (pixel_values.grad * pixel_values).float().sum(-3).sum(0).detach().cpu()
            else:
                raise ValueError(f"Unexpected pixel_values shape for {model_type}")

            rel_text = (text_embeds.grad * text_embeds).float().sum(-1).detach().cpu()

            # Shape: (1, ...)
            token_attributions.append(rel_text)
            pixel_attributions.append(rel_img)

        else:
            # --- TOKEN-WISE (ISOLATED) ---
            for idx in indices_to_compute:
                # Fresh tensors for isolated gradients
                pixel_values = base_pixel_values.clone().requires_grad_(True)
                text_embeds = base_text_embeds.clone().requires_grad_(True)

                self.wrapper.model.zero_grad()

                # Fresh forward pass prevents internal gradient accumulation
                log_logits = self.wrapper.get_captum_forward(
                    text_embeds=text_embeds,
                    pixel_values=pixel_values,
                    kwargs_dict=kwargs_dict,
                    return_full_sequence=True,
                )

                gen_logits = log_logits[:, t_start - 1 : t_end - 1, :]
                gathered_logits = gen_logits[0, torch.arange(gen_len), generated_token_ids]

                # Backpropagate just this specific token
                target_logit = gathered_logits[idx]
                target_logit.backward(retain_graph=False)  # False is critical here!

                # LRP Rule: Relevance = Gradient * Activation
                if "qwen" in model_type:
                    rel_img = (pixel_values.grad * pixel_values).float().sum(-1).detach().cpu()
                elif "internvl" in model_type:
                    rel_img = (pixel_values.grad * pixel_values).float().sum(-3).detach().cpu()
                elif "llava" in model_type:
                    rel_img = (pixel_values.grad * pixel_values).float().sum(-3).sum(0).detach().cpu()
                else:
                    raise ValueError(f"Unexpected pixel_values shape for {model_type}")

                rel_text = (text_embeds.grad * text_embeds).float().sum(-1).detach().cpu()

                token_attributions.append(rel_text)
                pixel_attributions.append(rel_img)

                del log_logits, pixel_values, text_embeds
                torch.cuda.empty_cache()

        # 3. CLEANUP & FORMATTING
        if self.zennit_comp is not None:
            self.zennit_comp.remove()
            self.zennit_comp = None

        torch.cuda.empty_cache()

        # # Stack into [len(target_indices), ...] tensors
        # if "llava" in model_type:
        #      # LLaVA already summed dim 0, so we just stack directly
        #      relevance_img = torch.stack(pixel_attributions, dim=0)
        # else:
        #      relevance_img = torch.cat(pixel_attributions, dim=0)
        relevance_img = torch.stack(pixel_attributions, dim=0)
        relevance_text = torch.cat(token_attributions, dim=0)

        # Slice the text relevance to only include the prompt
        relevance_text = relevance_text[:, :t_start]

        return relevance_text, relevance_img

