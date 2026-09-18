import argparse
import contextlib
import os
import time
import traceback
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import transformers
from packaging.version import Version
from PIL import ImageFilter
from torch import Tensor
from tqdm import tqdm

if Version(transformers.__version__) > Version("5.10"):
    pass

# --- ABSTRACTED FACTORIES & UTILS ---
from src.datasets.factory import get_dataloader
from src.explainers.factory import get_explainer
from src.metrics.shap_sii import eval_pure_sii_with_class
from src.models.factory import load_vlm
from src.utils.faithfulness_utils import (
    _reshape_pixels_back_faithfulness,
    _reshape_pixels_faithfulness,
    get_most_important_tokens_multimodal,
    make_blur_baseline,
    score_output,
)
from src.utils.xai_utils import find_ynvqa_token_index, get_processed_indices, load_yaml, save_to_jsonl


def build_object_semantic_mask(input_ids, tokenizer, object_name):
    """
    Dynamically finds the tokens corresponding to the 'object_name' within the full 
    tokenized prompt (including chat templates) and creates a boolean mask.
    
    Args:
        input_ids (Tensor): The tokenized input of shape (1, seq_len).
        tokenizer: The HuggingFace tokenizer associated with the VLM.
        object_name (str): The core object to isolate (e.g., "snowboard").
        
    Returns:
        Tensor: A boolean mask of shape (1, seq_len) where True indicates the object tokens.
    """
    device = input_ids.device
    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    ids_list = input_ids[0].tolist()
    
    # 1. Reconstruct the full string while tracking the character span of each token
    current_str = ""
    token_char_spans = []
    
    for tid in ids_list:
        # We decode each token individually to see its exact string representation
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        start_char = len(current_str)
        current_str += token_str
        end_char = len(current_str)
        
        token_char_spans.append((start_char, end_char, token_str))
        
    # 2. Find the character position of the object in the full reconstructed string
    target_start = current_str.lower().find(object_name.lower())
    
    if target_start == -1:
        print(f"[Warning] Object '{object_name}' not found in the prompt sequence!")
        # Fallback: just activate the last few tokens before special tokens if not found
        return mask 
        
    target_end = target_start + len(object_name)
    
    # 3. Flag any token whose character span overlaps with the object's character span
    for i, (start, end, text) in enumerate(token_char_spans):
        # Math check for overlapping intervals
        if max(start, target_start) < min(end, target_end) and text.strip() != "":
            # Exclude tokens that are purely whitespace (saves Shapley budget)
            mask[0, i] = True
                
    # Quick debug print so you can verify it caught the right subwords
    active_tokens = [tokenizer.decode([ids_list[i]]) for i in range(len(ids_list)) if mask[0, i]]
    print(f"[*] Object '{object_name}' mapped to tokens: {active_tokens} (Total: {len(active_tokens)})")
                
    return mask[0]

def safe_flatten_to_list(array_like):
    if hasattr(array_like, "flatten"):
        return array_like.flatten().tolist()
    elif hasattr(array_like, "tolist"):
        flat_list = []
        for item in array_like.tolist():
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        return flat_list
    return list(array_like)


@torch.no_grad()
def eval_multimodal_synergy_batch(
    model: Any,  # BaseVLMWrapper
    inputs: dict[str, Any],
    target_ids: Tensor,
    pixel_attribution: Tensor,  # (B, H, W)
    token_attribution: Tensor,  # (B, L_prompt)
    perturbation_steps: Sequence[float],
    pad_token_id: int,  # Text Baseline
    special_token_ids: list[int],  # For filtering text tokens
    semantic_mask: Tensor | None = None,
    blur_baseline: Tensor | None = None,  # Image Baseline (same shape as pixel_values)
    mask_value: float = 0.0,
    descending: bool = True,  # True = "Insertion" style (Start from 0, add Important)
    filter_keywords: bool = True,
) -> dict[str, Any]:
    """
    Computes the Synergy and PID Alignment between Image and Text attributions.
    Formula: P(Img, Txt) - (P(Img, 0) + P(0, Txt))
    Upgraded with Partial Information Decomposition (PID) to isolate Redundancy.
    """
    device = model.device
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    pixel_attribution = pixel_attribution.to(device)
    token_attribution = token_attribution.to(device)
    target_ids = target_ids.to(device)

    # --- SAFELY ADD BATCH DIMENSION BASED ON ARCHITECTURE ---
    model_type = getattr(model.model.config, "model_type", "").lower()

    if "internvl" in model_type:
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
    elif "qwen" in model_type:
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)
    else:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

    # ---------- normalize shapes & define feature/position dims for Image Inputs ----------
    origin_shape = pixel_values.shape
    model_type = model.model.config.model_type

    # # Setup Baselines & Flattening
    feat, num_pixels = _reshape_pixels_faithfulness(
        pixel_values=pixel_values, origin_shape=origin_shape, model_type=model_type
    )
    num_img_feat = feat.shape[1]

    # Baseline image (blur)
    if blur_baseline is None:
        blur_baseline = make_blur_baseline(pixel_values=pixel_values, model_type=model_type)
    feat_baseline = blur_baseline.clone().reshape(feat.shape)

    # Flatten Attribution Map
    B, *_ = origin_shape
    if pixel_attribution.ndim == 4 or pixel_attribution.ndim == 3:
        sal_flat_img = pixel_attribution.reshape(B, -1)
    elif pixel_attribution.ndim == 2:
        sal_flat_img = pixel_attribution
    else:
        raise ValueError("pixel_attribution must be 2D, 3D, or 4D.")

    # --- 2. Setup Text Inputs ---
    if semantic_mask is not None:
        if semantic_mask.ndim == 1:
            semantic_mask = semantic_mask.unsqueeze(0)
        valid_mask = semantic_mask.to(device).clone()
    else:
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)

    if special_token_ids is not None:
        for skip_id in special_token_ids:
            valid_mask &= input_ids != skip_id

    num_valid_tokens = valid_mask.sum(dim=1).min().item()

    # --- Mask Attribution Scores ---
    masked_attribution = token_attribution.clone()
    masked_attribution[~valid_mask] = -float("inf")

    # --- Baselines ---
    baseline_input_ids = input_ids.clone()
    baseline_input_ids[valid_mask] = pad_token_id

    # --- 3. Baselines & Targets ---
    if filter_keywords:
        target_positions = get_most_important_tokens_multimodal(
            model,
            inputs,
            input_ids,
            baseline_input_ids,
            pixel_values,
            blur_baseline,
            target_ids,
        )
    else:
        seq_len = target_ids.shape[1]
        default_indices = torch.arange(seq_len, device=device)
        target_positions = [default_indices for _ in range(B)]

    zeros_scores = score_output(
        model,
        inputs=inputs,
        input_ids=baseline_input_ids,
        pixel_values=blur_baseline,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()  # (B,)

    full_scores = score_output(
        model,
        inputs=inputs,
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()  # (B,)

    normalizer = np.maximum(np.abs(full_scores - zeros_scores), 0.05)

    # --- 4. Loop Initialization ---
    S = len(perturbation_steps)

    # Original Curves
    del_synergy_curve = np.zeros((S, B), dtype=np.float32)
    ins_synergy_curve = np.zeros((S, B), dtype=np.float32)
    del_norm_synergy_curve = np.zeros((S, B), dtype=np.float32)
    ins_norm_synergy_curve = np.zeros((S, B), dtype=np.float32)

    # NEW: PID Curves (Insertion)
    ins_redundancy_curve = np.zeros((S, B), dtype=np.float32)
    ins_true_syn_curve = np.zeros((S, B), dtype=np.float32)
    ins_alignment_curve = np.zeros((S, B), dtype=np.float32)

    # NEW: PID Curves (Deletion)
    del_redundancy_curve = np.zeros((S, B), dtype=np.float32)
    del_true_syn_curve = np.zeros((S, B), dtype=np.float32)
    del_alignment_curve = np.zeros((S, B), dtype=np.float32)

    for i, step in enumerate(perturbation_steps):
        # --- A. Determine K ---
        k_img = int(round(step * num_pixels))
        k_img = max(0, min(k_img, num_pixels))
        k_txt = int(round(step * num_valid_tokens))
        k_txt = int(max(0, min(k_txt, num_valid_tokens)))

        _, top_img_idx = torch.topk(sal_flat_img, k_img, dim=-1, largest=descending)
        top_img_idx_exp = top_img_idx.unsqueeze(1).expand(B, num_img_feat, k_img)

        _, top_token_idx = torch.topk(masked_attribution, k_txt, dim=-1, largest=descending)

        # --- B. Create Perturbed Modalities ---
        # Deletion Modalities
        feat_pixels = feat.clone()
        pixels_orig = feat_baseline.gather(dim=2, index=top_img_idx_exp)
        feat_pixels.scatter_(dim=2, index=top_img_idx_exp, src=pixels_orig)
        del_pixels = _reshape_pixels_back_faithfulness(
            feat_pert=feat_pixels, origin_shape=origin_shape, model_type=model_type
        )

        del_tokens = input_ids.clone()
        pad_src = torch.full_like(top_token_idx, pad_token_id)
        del_tokens.scatter_(dim=1, index=top_token_idx, src=pad_src)

        # Insertion Modalities
        feat_pert = feat_baseline.clone()
        mask_src = feat.gather(dim=2, index=top_img_idx_exp)
        feat_pert.scatter_(dim=2, index=top_img_idx_exp, src=mask_src)
        ins_pixels = _reshape_pixels_back_faithfulness(
            feat_pert=feat_pert, origin_shape=origin_shape, model_type=model_type
        )

        ins_tokens = baseline_input_ids.clone()
        orig_tokens = input_ids.gather(dim=1, index=top_token_idx)
        ins_tokens.scatter_(dim=1, index=top_token_idx, src=orig_tokens)

        # ==============================================================
        # DELETION SCORING & PID
        # ==============================================================
        del_p_joint = score_output(
            model,
            inputs,
            input_ids=del_tokens,
            pixel_values=del_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()
        del_p_img_only = score_output(
            model,
            inputs,
            input_ids=input_ids,
            pixel_values=del_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()
        del_p_txt_only = score_output(
            model,
            inputs,
            input_ids=del_tokens,
            pixel_values=pixel_values,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # Standard Deletion Interaction
        del_synergy = del_p_joint - (del_p_img_only + del_p_txt_only - full_scores)
        del_synergy_curve[i] = del_synergy
        del_norm_synergy_curve[i] = del_synergy / normalizer

        # --- NEW: Deletion PID Math ---
        # Redundancy: Min drop between removing just ONE modality vs BOTH
        r_del = np.maximum(0, np.minimum(del_p_img_only - del_p_joint, del_p_txt_only - del_p_joint))
        s_del = np.maximum(0, del_synergy + r_del)
        align_del = s_del + r_del

        del_redundancy_curve[i] = r_del
        del_true_syn_curve[i] = s_del
        del_alignment_curve[i] = align_del

        # ==============================================================
        # INSERTION SCORING & PID
        # ==============================================================
        ins_p_joint = score_output(
            model,
            inputs,
            input_ids=ins_tokens,
            pixel_values=ins_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()
        ins_p_img_only = score_output(
            model,
            inputs,
            input_ids=baseline_input_ids,
            pixel_values=ins_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()
        ins_p_txt_only = score_output(
            model,
            inputs,
            input_ids=ins_tokens,
            pixel_values=blur_baseline,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # Standard Insertion Interaction
        ins_synergy = ins_p_joint - (ins_p_img_only + ins_p_txt_only - zeros_scores)
        ins_synergy_curve[i] = ins_synergy
        ins_norm_synergy_curve[i] = ins_synergy / normalizer

        # --- NEW: Insertion PID Math ---
        # Information Gain from baseline
        gain_img = np.maximum(0, ins_p_img_only - zeros_scores)
        gain_txt = np.maximum(0, ins_p_txt_only - zeros_scores)

        # Redundancy: Min overlapping gain from either modality
        r_ins = np.minimum(gain_img, gain_txt)

        # True Synergy: Interaction with Redundancy added back
        s_ins = np.maximum(0, ins_synergy + r_ins)

        # Total Alignment (Faithfulness)
        align_ins = s_ins + r_ins

        ins_redundancy_curve[i] = r_ins
        ins_true_syn_curve[i] = s_ins
        ins_alignment_curve[i] = align_ins

    # --- Integrate Standard AUCs ---
    del_norm_syn_auc = np.trapezoid(del_norm_synergy_curve, x=perturbation_steps, axis=0)
    ins_norm_syn_auc = np.trapezoid(ins_norm_synergy_curve, x=perturbation_steps, axis=0)
    del_syn_auc = np.trapezoid(del_synergy_curve, x=perturbation_steps, axis=0)
    ins_syn_auc = np.trapezoid(ins_synergy_curve, x=perturbation_steps, axis=0)

    # --- Integrate NEW PID AUCs ---
    ins_redundancy_auc = np.trapezoid(ins_redundancy_curve, x=perturbation_steps, axis=0)
    ins_true_syn_auc = np.trapezoid(ins_true_syn_curve, x=perturbation_steps, axis=0)
    ins_alignment_auc = np.trapezoid(ins_alignment_curve, x=perturbation_steps, axis=0)

    del_redundancy_auc = np.trapezoid(del_redundancy_curve, x=perturbation_steps, axis=0)
    del_true_syn_auc = np.trapezoid(del_true_syn_curve, x=perturbation_steps, axis=0)
    del_alignment_auc = np.trapezoid(del_alignment_curve, x=perturbation_steps, axis=0)

    return {
        # Standard Outputs
        "zeros_baseline": zeros_scores,
        "full_baseline": full_scores,
        "del_synergy_curve": del_synergy_curve,
        "ins_synergy_curve": ins_synergy_curve,
        "ins_norm_synergy_curve": ins_norm_synergy_curve,
        "del_norm_synergy_curve": del_norm_synergy_curve,
        "del_norm_auc": del_norm_syn_auc,
        "ins_norm_auc": ins_norm_syn_auc,
        "del_auc": del_syn_auc,
        "ins_auc": ins_syn_auc,
        # NEW: PID Insertion Outputs
        "ins_redundancy_auc": ins_redundancy_auc,
        "ins_true_syn_auc": ins_true_syn_auc,
        "ins_alignment_auc": ins_alignment_auc,
        # NEW: PID Deletion Outputs
        "del_redundancy_auc": del_redundancy_auc,
        "del_true_syn_auc": del_true_syn_auc,
        "del_alignment_auc": del_alignment_auc,
    }


def run_evaluation(args):
    # 1. Load Configurations
    dataset_config = load_yaml(args.dataset_config)
    model_config = load_yaml(args.model_config)

    # --- AUTO-PATH RESOLUTION ---
    explainer_paths = []
    for exp in args.explainers:
        if exp.endswith(".yaml"):
            explainer_paths.append(exp)  # User provided a direct path
        else:
            explainer_paths.append(f"configs/explainers/{exp}.yaml")  # Auto-resolve!

    # Load them using the resolved paths
    explainer_configs = [load_yaml(path) for path in explainer_paths]

    # Setup Output Directory
    output_dir = os.path.join(
        args.output_dir, model_config["name"], f"{dataset_config['name']}_{dataset_config['pope_type']}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 2. Attention "Lookahead" Optimization
    needs_attention = any(cfg.get("requires_attention", False) for cfg in explainer_configs)
    attn_mode = "eager" if needs_attention else None
    print(f"[*] Attention Implementation set to: {attn_mode}")

    # 3. Load Model (ONCE per script execution)
    model_wrapper = load_vlm(
        model_config=model_config,
        attn_implementation=attn_mode,
        gpu_node=args.gpu_id,
        output_attentions=needs_attention,
    )
    # Store the config on the wrapper so LXT Explainer can reload the model if needed!
    model_wrapper.model_config = {
        "model_config": model_config,
        "attn_implementation": attn_mode,
        "gpu_node": args.gpu_id,
        "output_attentions": needs_attention,
    }

    print(f"[*] Loading Dataset: {dataset_config['name']}...")
    dl = get_dataloader(dataset_config)

    # 4. Metric Hyperparameters
    pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pert_steps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    device = model_wrapper.device
    tok = model_wrapper.processor.tokenizer
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    special_token_ids = model_wrapper.special_token_ids
    filter_keywords = True

    # ---------------------------------------------------------
    # OUTER LOOP: Iterate over Requested Explainers
    # ---------------------------------------------------------
    for explainer_path in explainer_paths:
        explainer = None

        try:
            # Load explainer dynamically, injecting model config for CAM layers
            explainer, explainer_name = get_explainer(explainer_path, model_wrapper, model_config)

            print(f"\n{'=' * 50}\n[*] Evaluating: {explainer_name} on {model_config['name']}\n{'=' * 50}")

            # Initialize Weights & Biases
            run_name = f"{model_config['name']}_{dataset_config['name']}_{dataset_config['pope_type']}_{explainer_name}"

            output_file = os.path.join(output_dir, f"{run_name}_results.jsonl")

            # =========================================================
            # --- RESUME LOGIC: Find Already Processed Samples ---
            # =========================================================
            processed_indices = get_processed_indices(
                output_file=output_file, total_dataset_len=len(dl), max_samples=args.max_samples
            )

            # ---------------------------------------------------------
            # EVALUATION LOOP
            # ---------------------------------------------------------
            for idx, sample in enumerate(tqdm(dl, desc=f"Evaluating {args.explainers}")):
                if args.max_samples is not None and idx >= args.max_samples:
                    break

                if idx in processed_indices:
                    continue

                try:
                    img = sample["image"]
                    if args.resize_img is not None:
                        img = img.resize((args.resize_img, args.resize_img))
                    question = sample["question"]
                    label = sample.get("label", "unknown")
                    image_id = sample.get("image_id", f"unknown_{idx}")

                    # Extract Masks (if using Oracle/Anti)
                    keyword = sample.get("object_name")
                    oracle_mask_2d = sample.get("pixel_oracle_mask")

                    # 1. Forward Pass
                    inputs = model_wrapper.get_inputs(img, question)
                    pred_results = model_wrapper.predict(inputs, return_logits=True)

                    # Get text only mask
                    # model_type = getattr(model_wrapper.model.config, "model_type", "").lower()
                    semantic_mask = build_object_semantic_mask(inputs["input_ids"],
                                           model_wrapper.processor.tokenizer,
                                            sample["object_name"])
                    
                    # 2. Identify the Decision Token
                    yes_no_tok_idx = find_ynvqa_token_index(
                        pred_results["new_ids"], text_answer=pred_results["text"], tokenizer=tok
                    )
                    if yes_no_tok_idx is None:
                        yes_no_tok_idx = 0

                    # 3. Generate Attributions
                    # Pass Oracle kwargs safely (TAM ignores them, Oracle uses them)
                    text_attrs, img_attrs = explainer.attribute(
                        img,
                        text=question,
                        target_indices=yes_no_tok_idx,
                        pred_results=pred_results,
                        keyword=keyword,
                        oracle_mask_2d=oracle_mask_2d,
                    )

                    # Slice strictly for the target token
                    pixel_attribution = img_attrs[0 : 0 + 1]
                    token_attribution = text_attrs[0 : 0 + 1]
                    target_ids = pred_results["new_ids"].unsqueeze(0)

                    # 4. Generate Image Baseline (Model Agnostic!)
                    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=30))
                    blur_inputs = model_wrapper.get_inputs(blurred_img, question)
                    blur_baseline = blur_inputs["pixel_values"].to(device)

                    log_dict = {
                        "sample_idx": idx,
                        "image_id": image_id,
                        "question": question,
                        "label": label,
                        "prediction": pred_results.get("text"),
                    }

                    # ==============================================================
                    # METRIC 1: EXACT SHAP SII (Computationally Heavy)
                    # ==============================================================
                    start_time = time.perf_counter()
                    shap_sii_res = eval_pure_sii_with_class(
                        model=model_wrapper,
                        inputs=inputs,
                        target_ids=target_ids,
                        token_attribution=token_attribution,
                        pixel_attribution=pixel_attribution,
                        perturbation_steps=pert_steps,
                        pad_token_id=pad_token_id,
                        special_token_ids=special_token_ids,
                        filter_keywords=filter_keywords,
                        blur_baseline=blur_baseline,
                        mask_value=0.0,
                        semantic_mask=semantic_mask,
                        #n_background_groups=args.n_background_groups,
                        #shapiq_budget=args.shapiq_budget,
                        batch_size=args.batch_size,
                    )
                    log_dict["time_sii"] = time.perf_counter() - start_time
                    log_dict["sii_auc"] = float(shap_sii_res["sii_auc"].item())

                    if "sii_curve" in shap_sii_res:
                        log_dict["sii_curve"] = safe_flatten_to_list(shap_sii_res["sii_curve"])

                    # ==============================================================
                    # METRIC 2: FAST PID SYNERGY (With Grammar Freeze)
                    # ==============================================================
                    start_time = time.perf_counter()
                    syn_res = eval_multimodal_synergy_batch(
                        model=model_wrapper,
                        inputs=inputs,
                        target_ids=target_ids,
                        token_attribution=token_attribution,
                        pixel_attribution=pixel_attribution,
                        perturbation_steps=pert_steps,
                        pad_token_id=pad_token_id,
                        special_token_ids=special_token_ids,
                        semantic_mask=semantic_mask,  # Passes the frozen grammar mask!
                        blur_baseline=blur_baseline,
                        mask_value=0.0,
                        filter_keywords=filter_keywords,
                    )
                    log_dict["time_syn"] = time.perf_counter() - start_time

                    for key, val in syn_res.items():
                        if "synergy_curve" in key:
                            log_dict[f"syn_{key}"] = safe_flatten_to_list(val)

                    # Compute Final Aggregated PID Scores
                    ins_align = float(syn_res["ins_alignment_auc"].item())
                    del_align = float(syn_res["del_alignment_auc"].item())
                    log_dict["final_alignment"] = (ins_align + (1.0 - del_align)) / 2.0

                    ins_true_syn = float(syn_res["ins_true_syn_auc"].item())
                    del_true_syn = float(syn_res["del_true_syn_auc"].item())
                    log_dict["final_true_syn"] = (ins_true_syn + (1.0 - del_true_syn)) / 2.0

                    ins_red = float(syn_res["ins_redundancy_auc"].item())
                    del_red = float(syn_res["del_redundancy_auc"].item())
                    log_dict["final_redundancy"] = (ins_red + (1.0 - del_red)) / 2.0

                    log_dict["standard_srg"] = float(syn_res["ins_norm_auc"].item()) - float(
                        syn_res["del_norm_auc"].item()
                    )
                    log_dict["synergy_srg"] = (
                        float(syn_res["ins_norm_auc"].item()) + float(syn_res["del_norm_auc"].item())
                    ) / 2.0
                    log_dict["synergy_refined_srg"] = (
                        float(syn_res["ins_norm_auc"].item()) + (1.0 - float(syn_res["del_norm_auc"].item()))
                    ) / 2.0

                    # Save
                    save_to_jsonl(log_dict, output_file)

                    # Cleanup to prevent OOM
                    del inputs, blur_inputs, text_attrs, img_attrs, syn_res
                    del shap_sii_res
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Failed on sample {idx}: {e}")
                    traceback.print_exc()
                    continue

        # --- THE NEW CRASH CATCHER ---
        # Cleanup
        except Exception as e:
            print(f"\n[!] ERROR: Explainer '{explainer_path}' crashed completely!")
            print(f"[!] Exception Details: {e}")
            print("[!] Skipping this explainer and moving to the next one...\n")

            # 2. Tell W&B that this specific run crashed, so it doesn't hang
            # if wandb.run is not None:
            #     wandb.finish(exit_code=1)

            # 3. Move on to the next explainer!
            continue

        finally:
            print(f"[*] Finished {explainer_path}. Cleaning up GPU memory...")
            # 1. Safely force-delete the explainer from VRAM if it initialized
            if explainer is not None:
                print("[*] Cleaning up GPU memory...")
                if hasattr(explainer, "cleanup"):
                    with contextlib.suppress(Exception):
                        explainer.cleanup()

                del explainer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified XAI Evaluation Suite")

    # Configs
    parser.add_argument("--model_config", type=str, required=True, help="Path to model YAML")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument(
        "--explainers",
        nargs="+",
        required=True,
        help="List of paths to explainer YAMLs",
    )
    parser.add_argument("--resize_img", type=int, default=32,)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU node to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/results_test_baseline",
        help="Where to save logs",
    )

    # Metric Params
    parser.add_argument("--shapiq_budget", type=int, default=1200, help="Budget for ShapIQ interaction approx")
    parser.add_argument("--n_background_groups", type=int, default=8, help="Grouping for SII pixels")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for metric scoring")

    args = parser.parse_args()

    try:
        run_evaluation(args)
    except Exception as e:
        print(f"[FATAL] Benchmark crashed: {e}")
