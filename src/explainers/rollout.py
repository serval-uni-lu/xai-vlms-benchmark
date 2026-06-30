import contextlib
import inspect
import torch
import torch.nn.functional as F

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from src.utils.xai_utils import align_llm_visuals_to_pixels


def stitch_chunks_to_matrix(chunk_list):
    """Helper to stitch block-diagonal chunks for Qwen-VL."""
    if not isinstance(chunk_list, list) or len(chunk_list) == 1:
        return chunk_list[0] if isinstance(chunk_list, list) else chunk_list

    batch_size, num_heads, _, _ = chunk_list[0].shape
    total_seq_len = sum(chunk.shape[2] for chunk in chunk_list)
    full_matrix = torch.zeros(
        (batch_size, num_heads, total_seq_len, total_seq_len),
        device=chunk_list[0].device,
        dtype=chunk_list[0].dtype,
    )
    current_idx = 0
    for chunk in chunk_list:
        chunk_len = chunk.shape[2]
        full_matrix[
            :,
            :,
            current_idx : current_idx + chunk_len,
            current_idx : current_idx + chunk_len,
        ] = chunk
        current_idx += chunk_len
    return full_matrix


class RolloutExplainer(BaseExplainer):
    def __init__(self, model_wrapper: BaseVLMWrapper, requires_grad=False):
        super().__init__(model_wrapper)
        self.model = self.wrapper.model
        self.requires_grad = requires_grad  # Set to True for Grad Rollout

        self.vision_attentions = []
        self.llm_attentions = []
        self.vision_grads = []
        self.llm_grads = []
        self.hooks = []

    def _generic_hook(self, module, output, attns_list, grads_list):
        """A single, smart hook to handle both Vision and LLM safely."""
        # 1. Safely extract the weights (Stashed OR from output tuple)
        attn_data = getattr(module, "saved_attn_weights", None)
        if attn_data is None:
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_data = output[1]
            else:
                # ==========================================================
                # QWEN 3.5 HYBRID ARCHITECTURE / MULTI-GPU FIX
                # If a layer (like a Gated DeltaNet) does not physically 
                # compute an attention matrix, or if accelerate stripped it,
                # we gracefully skip it and let Rollout calculate over the 
                # remaining Full Attention layers.
                # ==========================================================
                return


        # 2. Force it into a list so we can handle standard models and Qwen interchangeably
        chunk_list = attn_data if isinstance(attn_data, list) else [attn_data]

        if self.requires_grad:
            attns_list.append(chunk_list)
            layer_grads = [None] * len(chunk_list)

            def create_hook(idx, target_list):
                return lambda grad: target_list.__setitem__(idx, grad.clone().detach().cpu())

            for i, chunk in enumerate(chunk_list):
                if chunk is not None:
                    if not chunk.requires_grad:
                        print(f"[!] WARNING: Attention chunk {i} does not require grad!")
                    else:
                        chunk.retain_grad()
                        chunk.register_hook(create_hook(i, layer_grads))

            grads_list.append(layer_grads)
        else:
            # Detach and CPU offload to save VRAM
            cpu_chunks = [chunk.detach().cpu() for chunk in chunk_list]
            attns_list.append(cpu_chunks)

        # Cleanup stash to prevent memory leaks
        module.saved_attn_weights = None
    
    # def _force_attention_pre_hook(self, module, args, kwargs):
    #     """
    #     Surgically injects output_attentions=True into the kwargs 
    #     just milliseconds before the forward pass runs, bypassing multi-GPU wrappers.
    #     """
    #     sig = inspect.signature(module.forward)
    #     if "output_attentions" in sig.parameters:
    #         kwargs["output_attentions"] = True
    #     return args, kwargs

    def _vision_hook(self, module, input, output):
        self._generic_hook(module, output, self.vision_attentions, self.vision_grads)

    def _llm_hook(self, module, input, output):
        self._generic_hook(module, output, self.llm_attentions, self.llm_grads)

    def register_hooks(self):
        self.clear_hooks()
        for _, module in self.model.named_modules():
            if self.wrapper.vision_module_name and self.wrapper.vision_module_name in str(type(module)):
                self.hooks.append(module.register_forward_hook(self._vision_hook))
            elif self.wrapper.llm_module_name and self.wrapper.llm_module_name in str(type(module)):
                self.hooks.append(module.register_forward_hook(self._llm_hook))
    
    # def register_hooks(self):
    #     self.clear_hooks()
    #     for _, module in self.model.named_modules():
    #         # --- VISION HOOKS ---
    #         if self.wrapper.vision_module_name and self.wrapper.vision_module_name in str(type(module)):
    #             # 1. Inject the flag BEFORE the layer runs
    #             self.hooks.append(module.register_forward_pre_hook(self._force_attention_pre_hook, with_kwargs=True))
    #             # 2. Catch the matrix AFTER the layer runs
    #             self.hooks.append(module.register_forward_hook(self._vision_hook))
                
    #         # --- LLM HOOKS ---
    #         elif self.wrapper.llm_module_name and self.wrapper.llm_module_name in str(type(module)):
    #             # Do the same for LLM layers to be bulletproof
    #             self.hooks.append(module.register_forward_pre_hook(self._force_attention_pre_hook, with_kwargs=True))
    #             self.hooks.append(module.register_forward_hook(self._llm_hook))

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.vision_attentions.clear()
        self.llm_attentions.clear()
        self.vision_grads.clear()
        self.llm_grads.clear()
        torch.cuda.empty_cache()

    @contextlib.contextmanager
    def manage_explainability_state(self):
        """
        Temporarily patches the model and attaches hooks.
        Guarantees complete restoration upon exit.
        """
        # Save the original blueprint and apply the patch
        self.wrapper.apply_patch()

        # Register all PyTorch hooks
        self.register_hooks()

        try:
            # Yield control back to the main function
            yield
        finally:
            # This runs NO MATTER WHAT (even if your math throws an error)
            self.clear_hooks()

            # Restore the original Hugging Face blueprint
            self.wrapper.remove_patch()

    def _compute_rollout_math(self, attentions, gradients=None) -> torch.Tensor:
        """The core mathematical engine for Rollout with aggressive VRAM management."""
        if not attentions:
            raise ValueError("Need attention weights to compute Attention (Grad)Rollout")

        math_device = getattr(self, "device", "cuda")

        # Calculate total sequence length based on the chunks in Layer 0
        total_seq_len = sum(chunk.shape[2] for chunk in attentions[0])
        rollout = torch.eye(total_seq_len, device=math_device)

        if gradients is not None and len(gradients) > 0:
            # GRADIENT ROLLOUT (Chefer et al. 2021)

            for layer_attns, layer_grads in zip(attentions, gradients, strict=False):
                processed_chunks = []

                # --- OOM FIX: DO THE MATH ON TINY CHUNKS BEFORE STITCHING ---
                for a, g in zip(layer_attns, layer_grads, strict=False):
                    # Move just this tiny chunk to GPU
                    a_gpu = a.to(math_device)
                    g_gpu = g.to(math_device) if g is not None else torch.zeros_like(a_gpu)

                    # Multiply and ReLU
                    grad_attn_chunk = a_gpu * g_gpu
                    grad_attn_chunk = torch.clamp(grad_attn_chunk, min=0.0)

                    # Collapse heads immediately! (e.g., 32 heads -> 1 head)
                    # keepdim=True maintains (B, 1, seq, seq) shape so the stitcher doesn't break
                    grad_attn_chunk = grad_attn_chunk.mean(dim=1, keepdim=True)
                    processed_chunks.append(grad_attn_chunk)

                    # Free chunk VRAM immediately
                    del a_gpu, g_gpu

                # --- STITCH THE COLLAPSED CHUNKS ---
                # This matrix is now 32x smaller in VRAM!
                stitched_grad_attn = stitch_chunks_to_matrix(processed_chunks)

                # Remove batch and head dims -> shape: (total_seq_len, total_seq_len)
                grad_attn = stitched_grad_attn.squeeze(0).squeeze(0)

                grad_attn = grad_attn + torch.eye(total_seq_len, device=math_device)
                rollout = torch.matmul(grad_attn, rollout)

                del stitched_grad_attn, grad_attn
                torch.cuda.empty_cache()

        else:
            # STANDARD ROLLOUT (Abnar et al. 2020)
            for layer_attns in attentions:
                processed_chunks = []

                for a in layer_attns:
                    a_gpu = a.to(math_device)
                    # Collapse heads immediately!
                    attn_chunk = a_gpu.mean(dim=1, keepdim=True)
                    processed_chunks.append(attn_chunk)
                    del a_gpu

                stitched_attn = stitch_chunks_to_matrix(processed_chunks)
                attn_fused = stitched_attn.squeeze(0).squeeze(0)

                attn_fused = attn_fused + torch.eye(total_seq_len, device=math_device)
                attn_fused = torch.nn.functional.normalize(attn_fused, p=1, dim=-1)

                rollout = torch.matmul(attn_fused, rollout)

                del stitched_attn, attn_fused
                torch.cuda.empty_cache()

        return rollout

    def _attribute(
        self,
        image,
        text,
        target_indices: int | list[int] | None = None,
        average=False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the model, extracts attentions, and returns Vision and LLM rollouts."""

        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(
                inputs=inputs,
                return_logits=False,
                **kwargs,
            )
        full_ids = pred_results["full_ids"].to(self.device)

        # Define the indices of the answers tokens and visual tokens
        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]
        num_ans_tokens = t_end - t_start

        # --- DYNAMIC INDICES RESOLUTION ---
        if target_indices is None:
            indices_to_compute = list(range(num_ans_tokens))
        elif isinstance(target_indices, int):
            indices_to_compute = [target_indices]
        else:
            indices_to_compute = target_indices

        # Safety check
        indices_to_compute = [idx for idx in indices_to_compute if idx < num_ans_tokens]

        inputs["input_ids"] = full_ids.clone().unsqueeze(0)  # (batch, seq_len)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # --- MULTIMODAL ROPE FIX FOR QWEN ---
        # If the model requires mm_token_type_ids, we must expand it to match the new input_ids length.
        # Since all generated tokens are text, we pad it with the text type (0).
        if "mm_token_type_ids" in inputs:
            orig_mm_types = inputs["mm_token_type_ids"]
            
            # Create a tensor of 0s (text type) for the newly generated answer tokens
            pad_types = torch.zeros(
                (orig_mm_types.shape[0], num_ans_tokens), 
                dtype=orig_mm_types.dtype, 
                device=orig_mm_types.device
            )
            
            # Concatenate them together so it matches the length of the full_ids
            inputs["mm_token_type_ids"] = torch.cat([orig_mm_types, pad_types], dim=1)
            
        # Optional: Also catch position_ids if they were explicitly passed by the processor
        if "position_ids" in inputs:
            # We delete it so Qwen is forced to dynamically recalculate the correct 
            # position IDs using the newly padded mm_token_type_ids
            del inputs["position_ids"]

        with self.manage_explainability_state():
            if self.requires_grad:  # GradientxRollout
                outputs = self.model(**inputs, output_attentions=True)
                logits = outputs.logits[:, t_start - 1 : t_end - 1, :]  # (1, num_ans_tokens, vocab_size)

                new_ids = full_ids[t_start:t_end]
                new_ids = new_ids.unsqueeze(0).unsqueeze(-1)

                target_logits = logits.gather(dim=-1, index=new_ids).squeeze(-1)  # (1, num_ans_tokens)

                # --- TARGETED GRADIENT FIX ---
                # Only compute gradients for the specific tokens we care about!
                answer_score = target_logits[0, indices_to_compute].sum()

                self.model.zero_grad()
                answer_score.backward(retain_graph=False)

                v_rollout = self._compute_rollout_math(self.vision_attentions, self.vision_grads)
                t_rollout = self._compute_rollout_math(self.llm_attentions, self.llm_grads)
                v_rollout = v_rollout.detach().cpu()
                t_rollout = t_rollout.detach().cpu()
            else:  # Standard Rollout
                with torch.no_grad():
                    _ = self.model(**inputs, output_attentions=True)
                v_rollout = self._compute_rollout_math(self.vision_attentions)
                t_rollout = self._compute_rollout_math(self.llm_attentions)

        ## Compute the final attribution
        image_token_id = self.model.config.image_token_id
        is_image_mask = full_ids == image_token_id
        is_image_mask = is_image_mask.cpu()
        # is_text_mask = ~is_image_mask

        prompt_mask = (
            torch.arange(
                full_ids.size(-1),
            )
            < t_start
        )
        # final_text_mask = is_text_mask & prompt_mask
        final_image_mask = is_image_mask & prompt_mask

        # --- SUBSETTING FIX ---
        # Map the relative answer indices to absolute row positions in the rollout matrix
        abs_target_indices = [t_start - 1 + idx for idx in indices_to_compute]

        # Token attribution: (num_targets, num_prompt_tokens)
        token_attribution = t_rollout[abs_target_indices][:, prompt_mask]

        if average:
            token_attribution = token_attribution.mean(dim=0)

        # Image attribution: (num_targets, num_image_tokens)
        text_to_image_attn = t_rollout[abs_target_indices][:, final_image_mask]

        model_type = self.wrapper.model.config.model_type.lower()

        if "internvl" in model_type:
            pixel_values = inputs.get("pixel_values")
            pixel_attribution = self._fuse_and_upsample_internvl(
                text_to_image_attn=text_to_image_attn,
                v_rollout=v_rollout,
                pixel_values=pixel_values,
            )

        elif "qwen" in model_type:
            text_to_vit_attn = align_llm_visuals_to_pixels(text_to_image_attn, inputs, config=self.wrapper.model.config)
            pixel_attribution = torch.matmul(text_to_vit_attn, v_rollout)

        elif "llava" in model_type:
            if v_rollout.shape[-1] == text_to_image_attn.shape[-1] + 1:
                v_rollout = v_rollout[1:, 1:]

            patch_attribution = torch.matmul(text_to_image_attn, v_rollout)
            pixel_attribution = align_llm_visuals_to_pixels(patch_attribution, inputs, config=self.wrapper.model.config)

        else:
            raise NotImplementedError(f"This model {model_type} is not yet implemented for Rollout !")

        if average:
            pixel_attribution = pixel_attribution.mean(dim=0)

        self.clear_hooks()
        return token_attribution, pixel_attribution

    def _fuse_and_upsample_internvl(self, text_to_image_attn, v_rollout, pixel_values):
        """
        Dedicated Rollout fusion strictly for InternVL.
        InternVL compresses 1024 ViT patches (32x32) into 256 LLM tokens (16x16) per tile.
        """
        gen_len = text_to_image_attn.shape[0]

        # --- THE FIX: SQUEEZE DUMMY DIMENSIONS ---
        # If the shape is (num_tiles, 1, 1025, 1025) or (1, num_tiles, 1025, 1025),
        # we squeeze out the dimension of size 1 to make it properly 3D.
        if v_rollout.ndim == 4:
            if v_rollout.shape[1] == 1:
                v_rollout = v_rollout.squeeze(1)
            elif v_rollout.shape[0] == 1:
                v_rollout = v_rollout.squeeze(0)

        ndim = v_rollout.ndim

        # 1. Determine Tile Count & Strip the CLS token from ViT Rollout
        if ndim == 3:
            # 3D case: (num_tiles, 1025, 1025) -> (num_tiles, 1024, 1024)
            num_tiles = v_rollout.shape[0]
            if v_rollout.shape[1] == 1025:
                v_rollout = v_rollout[:, 1:, 1:]

        elif ndim == 2:
            # 2D case: (1025, 1025) -> (1, 1024, 1024)
            num_tiles = 1
            if v_rollout.shape[0] == 1025:
                v_rollout = v_rollout[1:, 1:]

            # Add the batch/tile dimension so torch.bmm works gracefully below
            v_rollout = v_rollout.unsqueeze(0)

        else:
            print(v_rollout.shape)
            print(text_to_image_attn.shape)
            raise ValueError(f"v_rollout ndim not implemented. Expected 2D or 3D, got {ndim}D.")

        # 3. Reshape LLM attention to the 16x16 latent grid
        # text_to_image_attn is currently (gen_len, num_tiles * 256)
        llm_attn_2d = text_to_image_attn.view(gen_len * num_tiles, 1, 16, 16)

        # 4. Upsample LLM attention to 32x32 to match the ViT patches!
        llm_attn_vit_res = F.interpolate(llm_attn_2d, size=(32, 32), mode="nearest")

        # 5. Prepare for Matrix Multiplication
        # Flatten to (gen_len, num_tiles, 1024) -> transpose to (num_tiles, gen_len, 1024)
        text_to_vit_attn = llm_attn_vit_res.view(gen_len, num_tiles, 1024).transpose(0, 1)

        # 6. Rollout Math: [num_tiles, gen_len, 1024] @ [num_tiles, 1024, 1024]
        # This works for both cases now because v_rollout is strictly 3D!
        pixel_attribution_patch = torch.bmm(text_to_vit_attn, v_rollout)

        # Transpose back to [gen_len, num_tiles, 1024]
        pixel_attribution_patch = pixel_attribution_patch.transpose(0, 1)

        # 7. Final Upsample to Raw Pixels (448x448)
        # Safely extract target height and width
        if pixel_values.ndim == 5:
            target_h, target_w = pixel_values.shape[3], pixel_values.shape[4]
        elif pixel_values.ndim == 4:
            target_h, target_w = pixel_values.shape[2], pixel_values.shape[3]
        else:
            target_h, target_w = 448, 448  # Safe InternVL default

        pixel_attr_2d = pixel_attribution_patch.reshape(gen_len * num_tiles, 1, 32, 32)

        pixel_attr_upscaled = F.interpolate(
            pixel_attr_2d,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # 8. Format output to exactly what the perturbation metric expects
        # Shape: (gen_len, num_tiles, H, W)
        return pixel_attr_upscaled.view(gen_len, num_tiles, target_h, target_w)
