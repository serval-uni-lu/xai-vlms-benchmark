import argparse
import contextlib
import os
import re
import time
import traceback

import numpy as np
import torch
import transformers
from packaging.version import Version
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

if Version(transformers.__version__) > Version("5.10"):
    pass

# from src.datasets.factory import get_dataloader
from src.datasets.gridvqa import GridQAMatchedDataset
from src.explainers.factory import get_explainer
from src.explainers.vqa_oracle import GridVQAAntiOracleExplainer, GridVQAOracleExplainer
from src.metrics import FaithfulnessMetric

# Import your custom Toy Model and Oracle Explainers
from src.models import LlavaWrapper
from src.models.factory import load_vlm
from src.models.toy_vlm import ToyGridVLM
from src.utils.xai_utils import get_processed_indices, load_yaml, save_to_jsonl

# =====================================================================
# 1. ANTI-ORACLE & PROXIMITY EVALUATOR
# =====================================================================

def get_dataloader(model_wrapper):
    dataset = GridQAMatchedDataset(
                        processor=model_wrapper.processor,
                        data_root="data/gridvqa/density30_pure",
                        split="val",
                        category=None,
                        #category="A",
                        )

    print(len(dataset))

    def collate_fn_custom(batch):
        return batch[0] if batch_size == 1 else batch

    batch_size = 1
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    collate_fn=collate_fn_custom)
    return dl




# Added "binary" and expanded common concepts
VQA_DOMAINS = {
    "binary": ["yes", "no"],
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "black",
              "white", "gray", "brown", "pink"],
    "shape": ["triangle", "square", "rectangle", "pentagon", "hexagon",
              "circle", "star", "cross", "cylinder", "cube", "sphere"],
    "count_words": [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", 
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", 
        "none"
    ]
}

def extract_predicted_concept(generated_text: str, label: str) -> str:
    """
    Finds the exact word/number the model predicted based on the ground truth's domain.
    Supports Yes/No, Colors, Shapes, and infinite integer digits.
    """
    gen_text_lower = generated_text.lower()
    label_lower = str(label).lower().strip()
    
    # 1. Figure out what domain the question belongs to
    target_domain = None
    
    # If the label is a pure number (e.g., "5", "12"), it's a count domain
    if label_lower.isdigit() or label_lower in VQA_DOMAINS["count_words"]:
        target_domain = "count"
    else:
        # Otherwise, check if it's binary, color, or shape
        for domain, vocab in VQA_DOMAINS.items():
            if label_lower in vocab:
                target_domain = domain
                break
                
    if target_domain is None:
        # Fallback: Just look for the exact label string
        return label_lower if label_lower in gen_text_lower else None

    # Tokenize the generated text into discrete words to avoid substring matches
    words = re.findall(r'\b\w+\b', gen_text_lower)
    
    # 2. Extract the predicted concept
    if target_domain == "count":
        # First, see if the model used a digit (e.g., "4")
        for word in words:
            if word.isdigit():
                return word
        # Second, check if it spelled the number out (e.g., "four")
        for word in words:
            if word in VQA_DOMAINS["count_words"]:
                return word
                
    elif target_domain in ["binary", "color", "shape"]:
        vocab = VQA_DOMAINS[target_domain]
        for word in words:
            if word in vocab:
                return word
                
    return None # The model mumbled or avoided answering the specific question

def parse_decision_token(
    new_ids: torch.Tensor, 
    tokenizer, 
    expected_label: str
) -> tuple[int, str|None]:
    """
    Finds the exact token index in the generated output that corresponds
    to the expected label, bypassing BPE token fragmentation issues.
    """
    if new_ids.dim() > 1:
        new_ids = new_ids[0]
        
    ids_list = new_ids.tolist()
    
    # 1. Reconstruct string and track character spans per token
    current_str = ""
    token_char_spans = []
    
    for tid in ids_list:
        token_str = tokenizer.decode([tid], skip_special_tokens=True)
        start_char = len(current_str)
        current_str += token_str
        end_char = len(current_str)
        token_char_spans.append((start_char, end_char, tid, token_str))
        
    gen_text_lower = current_str.lower()
    target_lower = str(expected_label).lower().strip()
    
    # 2. Locate the exact character span of the label in the generated text
    target_start_idx = gen_text_lower.find(target_lower)
    
    if target_start_idx == -1:
        # The model completely failed to generate the expected label
        return -1, None
        
    target_end_idx = target_start_idx + len(target_lower)
    
    # 3. Find the *first* meaningful token that falls inside this character span
    for i, (start, end, _tid, text) in enumerate(token_char_spans):
        # Check for intersection between token span and target span
        if max(start, target_start_idx) < min(end, target_end_idx) \
            and text.strip() != "":  # Ignore purely whitespace tokens
            return i, text
                
    return -1, None


def evaluate_explainer_proximity(oracle_attr: torch.Tensor,
                                explainer_attr: torch.Tensor) -> dict[str, float]:
    """Measures how closely a real heatmap matches the Oracle mask."""
    y_true = oracle_attr.detach().cpu().flatten().numpy()
    y_scores = explainer_attr.detach().cpu().flatten().numpy()
    
    try:
        ap_score = average_precision_score(y_true, y_scores)
    except ValueError:
        ap_score = 0.0

    K = int(y_true.sum())
    if K == 0 or len(y_true) == K:
        iou = 0.0
    else:
        top_k_indices = np.argsort(y_scores)[-K:]
        explainer_binary = np.zeros_like(y_true)
        explainer_binary[top_k_indices] = 1.0
        
        intersection = np.logical_and(y_true, explainer_binary).sum()
        union = np.logical_or(y_true, explainer_binary).sum()
        iou = intersection / union if union > 0 else 0.0

    return {"average_precision": float(ap_score), "top_k_iou": float(iou)}

# =====================================================================
# 2. MAIN BENCHMARK LOOP
# =====================================================================

def run_benchmark(args):
    # dataset_config = load_yaml(args.dataset_config)
    dataset_name = args.dataset_name
    model_config = load_yaml(args.model_config)

    explainer_paths = [exp if exp.endswith(".yaml") \
                       else f"configs/explainers/{exp}.yaml" for exp in args.explainers]
    explainer_configs = [load_yaml(path) for path in explainer_paths]

    output_dir = os.path.join(args.output_dir, model_config["vlm_type"], f"{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    needs_attention = any(cfg.get("requires_attention", False) for cfg in explainer_configs)
    attn_mode = "eager" if needs_attention else None

    gpu_id = args.gpu_id if args.gpu_id != -1 else "auto"

    # --- A. LOAD REAL MODEL ---
    model_wrapper = load_vlm(
        model_config=model_config,
        attn_implementation=attn_mode,
        gpu_node=gpu_id,
        output_attentions=needs_attention,
    )

    model_wrapper.model_config = {
            "model_config": model_config,
            "attn_implementation": attn_mode,
            "gpu_node": gpu_id,
            "output_attentions": needs_attention,
        }
    
    # --- B. LOAD TOY MODEL (For Evaluation) ---
    # We initialize it here once, and update its Ground Truth dynamically per sample
    toy_model = ToyGridVLM(model_wrapper=model_wrapper, label="dummy") # Label updated per sample
    toy_wrap = LlavaWrapper(model=toy_model, processor=model_wrapper.processor)

    # --- C. LOAD ORACLES ---
    # We pass the REAL wrapper to the Oracle so it builds masks matching the real model's tokenizer
    oracle_explainer = GridVQAOracleExplainer(model_wrapper)
    anti_oracle_explainer = GridVQAAntiOracleExplainer(model_wrapper)

    dl = get_dataloader(model_wrapper)

    pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    faith_metrics = FaithfulnessMetric(
        perturbation_steps=pert_steps,
        pad_token_id=model_wrapper.processor.tokenizer.pad_token_id or 0,
        special_token_ids=model_wrapper.special_token_ids,
        mask_value=0.0,
        filter_keywords=True,
    )

    for explainer_path in explainer_paths:
        explainer = None
        try:
            # --- D. LOAD REAL EXPLAINER ---
            explainer, explainer_name = get_explainer(explainer_path, model_wrapper, model_config)
            print(f"\n{'=' * 50}\n[*] Evaluating: {explainer_name}\n{'=' * 50}")

            run_name = f"{model_config['vlm_type']}_{dataset_name}_{explainer_name}"
            output_file = os.path.join(output_dir, f"{run_name}_results.jsonl")

            processed_indices = get_processed_indices(output_file, len(dl), args.max_samples)

            # ---------------------------------------------------------
            # INNER LOOP: Evaluate Dataset Samples
            # ---------------------------------------------------------
            for idx, sample in enumerate(tqdm(dl, desc=f"Evaluating {explainer_name}")):
                if args.max_samples is not None and idx >= args.max_samples:
                    break
                if idx in processed_indices:
                    continue

                #constraint = "" if "qwen" in model_config["vlm_type"] else "\nAnswer"
                constraint = ""
                img, question = sample["image"], sample["question"] + constraint
                label = sample["label"]

                try:
                    # 1. Forward Pass on REAL Model
                    inputs = model_wrapper.get_inputs(img, question)
                    pred_results = model_wrapper.predict(inputs, return_logits=True)
                    full_text = pred_results.get("text", "")

                    # 2. Figure out what the model ACTUALLY predicted (e.g. "four")
                    predicted_concept = extract_predicted_concept(full_text, label)
                    
                    if predicted_concept is None:
                        # Model mumbled nonsense and didn't give a recognizable shape/color/number
                        # Log as parse failure and skip
                        continue
                        
                    # 3. Find the exact token index of that concept in the sequence
                    decision_idx, decision_token_str = parse_decision_token(
                        new_ids=pred_results["new_ids"],
                        tokenizer=model_wrapper.processor.tokenizer,
                        expected_label=predicted_concept  # Look for "four", not "5"!
                    )

                    if decision_idx == -1:
                        decision_idx = 0
                        continue # BPE edge-case failure, skip
                    

                    # =======================================================
                    # PHASE 1: GENERATE ORACLE MASKS & CONFIGURE TOY MODEL
                    # =======================================================
                    oracle_kwargs = {
                        "text": question, 
                        "target_indices": [decision_idx], 
                        "pred_results": pred_results, 
                        "grid_metadata": sample["metadata"]
                    }
                    
                    oracle_txt, oracle_img = oracle_explainer.attribute(img, **oracle_kwargs)
                    anti_txt, anti_img = anti_oracle_explainer.attribute(img, **oracle_kwargs)

                    # Update Toy Model's Ground Truth State for THIS sample
                    target_id = pred_results["new_ids"][decision_idx].item()
                    toy_wrap.model.target_ids = [target_id] # Update target
                    toy_wrap.model.set_sample_ground_truth(
                        pixel_mask=sample["metadata"]["model_pixel_mask"],
                        text_mask=(oracle_txt[0] > 0.5), # Perfect 607-length mask!
                        orig_pixel_values=inputs["pixel_values"],
                        target_ids=[target_id]
                    )

                    # Get Theoretical Bounds for this sample
                    faith_sample = {"image": img, "text": question}
                    
                    oracle_faith = faith_metrics.compute(toy_wrap, faith_sample, 
                        {"inputs": inputs, "target_ids": pred_results["new_ids"].unsqueeze(0),
                         "pixel_attribution": oracle_img, "token_attribution": oracle_txt},
                        compute_syn_only=False)
                    
                    anti_faith = faith_metrics.compute(toy_wrap, faith_sample, 
                        {"inputs": inputs, "target_ids": pred_results["new_ids"].unsqueeze(0),
                        "pixel_attribution": anti_img, "token_attribution": anti_txt},
                        compute_syn_only=False)


                    # =======================================================
                    # PHASE 2: GENERATE REAL HEATMAPS & EVALUATE PROXIMITY
                    # =======================================================
                    start_time = time.perf_counter()
                    real_txt, real_img = explainer.attribute(img, text=question,
                                                            target_indices=[decision_idx],
                                                            pred_results=pred_results)

                    xai_gen_time = time.perf_counter() - start_time

                    # Calculate Proximity (Real vs Oracle)
                    img_prox = evaluate_explainer_proximity(oracle_img[0], real_img[0])
                    txt_prox = evaluate_explainer_proximity(oracle_txt[0], real_txt[0])

                    # Calculate F_syn for the Real Explainer
                    xai_real_result = {
                        "inputs": inputs,
                        "target_ids": pred_results["new_ids"].unsqueeze(0),
                        "pixel_attribution": real_img[0:1],
                        "token_attribution": real_txt[0:1]
                        }
                    real_faith = faith_metrics.compute(toy_wrap, faith_sample,
                                                       xai_result=xai_real_result
                        )

                    # =======================================================
                    # 3. CONSOLIDATE AND LOG RESULTS
                    # =======================================================
                    log_dict = {
                        "sample_idx": idx,
                        "explainer": explainer_name,
                        "question": question,
                        "label": label,
                        "prediction": full_text,
                        "predicted_concept": predicted_concept,
                        #"decision_token_str": decision_token_str,
                        "xai_gen_time": xai_gen_time,
                        "category": sample["category"],
                        
                        # Phase 1: Bounding Data
                        "oracle_syn_auc": oracle_faith.get("syn_synergy_norm_auc", 0.0),
                        "oracle_img_norm_auc_del": oracle_faith.get("img_norm_auc_del", 0.0),
                        "oracle_img_norm_auc_ins": oracle_faith.get("img_norm_auc_ins", 0.0),
                        "oracle_tok_norm_auc_del": oracle_faith.get("tok_norm_auc_del", 0.0),
                        "oracle_tok_norm_auc_ins": oracle_faith.get("tok_norm_auc_ins", 0.0),

                        "anti_oracle_syn_auc": anti_faith.get("syn_synergy_norm_auc", 0.0),
                        "anti_oracle_img_norm_auc_del": anti_faith.get("img_norm_auc_del", 0.0),
                        "anti_oracle_img_norm_auc_ins": anti_faith.get("img_norm_auc_ins", 0.0),
                        "anti_oracle_tok_norm_auc_del": anti_faith.get("tok_norm_auc_del", 0.0),
                        "anti_oracle_tok_norm_auc_ins": anti_faith.get("tok_norm_auc_ins", 0.0),
                        
                        # Phase 2: Proximity Data
                        "proximity_img_ap": img_prox["average_precision"],
                        "proximity_img_iou": img_prox["top_k_iou"],
                        "proximity_txt_ap": txt_prox["average_precision"],
                        "proximity_txt_iou": txt_prox["top_k_iou"],
                    }
                    
                    # Add Real Explainer's Faithfulness Scores
                    log_dict.update(real_faith)
                    
                    save_to_jsonl(log_dict, output_file)
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Explainer {explainer_name} failed on sample {idx}: {e}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"\n[!] ERROR: Explainer crashed completely! Details: {e}")
            continue
        finally:
            if explainer is not None and hasattr(explainer, "cleanup"):
                with contextlib.suppress(Exception):
                    explainer.cleanup()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM XAI Benchmark")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--explainers", nargs="+", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="logs/results_test_baseline")

    args = parser.parse_args()
    try:
        run_benchmark(args)
    except Exception as e:
        print(f"[FATAL] Benchmark crashed: {e}")
