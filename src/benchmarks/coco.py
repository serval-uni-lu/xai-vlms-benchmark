import argparse
import contextlib
import os
import time

import torch
from tqdm import tqdm

# --- ABSTRACTED FACTORIES & UTILS ---
from src.datasets.factory import get_dataloader
from src.explainers.factory import get_explainer
from src.metrics import FaithfulnessMetric, PlausibilityMetric
from src.models.factory import load_vlm
from src.utils.plausibility_utils import OntologyMapper, ids_to_word_groups
from src.utils.xai_utils import get_processed_indices, load_yaml, save_to_jsonl


def run_benchmark(args):
    # 1. Load Configurations
    dataset_config = load_yaml(args.dataset_config)
    model_config = load_yaml(args.model_config)

    # --- AUTO-PATH RESOLUTION ---
    explainer_paths = []
    for exp in args.explainers:
        if exp.endswith(".yaml"):
            explainer_paths.append(exp)
        else:
            explainer_paths.append(f"configs/explainers/{exp}.yaml")

    explainer_configs = [load_yaml(path) for path in explainer_paths]

    output_dir = os.path.join(args.output_dir, model_config["name"], dataset_config["name"])
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
    model_wrapper.model_config = {
        "model_config": model_config,
        "attn_implementation": attn_mode,
        "gpu_node": args.gpu_id,
        "output_attentions": needs_attention,
    }

    # 4. Load Dataset
    dl = get_dataloader(dataset_config)

    # Setup Ontology Mapper for Plausibility (Assume dl.dataset has id2name)
    category_dict = getattr(dl.dataset, "id2name", {})
    mapper = OntologyMapper(coco_categories=category_dict, threshold=0.5, device=model_wrapper.device)

    # 5. Initialize Metrics
    pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tok = model_wrapper.processor.tokenizer
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    faith_metrics = FaithfulnessMetric(
        perturbation_steps=pert_steps,
        pad_token_id=pad_token_id,
        special_token_ids=model_wrapper.special_token_ids,
        mask_value=0.0,
        filter_keywords=True,
    )

    plaus_metrics = PlausibilityMetric(ontology_mapper=mapper, category_dict=category_dict)

    # ---------------------------------------------------------
    # OUTER LOOP: Iterate over Requested Explainers
    # ---------------------------------------------------------
    for explainer_path in explainer_paths:
        explainer = None

        try:
            explainer, explainer_name = get_explainer(explainer_path, model_wrapper, model_config)

            print(f"\n{'=' * 50}\n[*] Evaluating: {explainer_name} on {model_config['name']}\n{'=' * 50}")

            run_name = f"{model_config['name']}_{dataset_config['name']}_{explainer_name}"
            # wandb.init(...)

            output_file = os.path.join(output_dir, f"{run_name}_results.jsonl")

            # =========================================================
            # --- RESUME LOGIC: Find Already Processed Samples ---
            # =========================================================
            processed_indices = get_processed_indices(
                output_file=output_file, total_dataset_len=len(dl), max_samples=args.max_samples
            )

            # ---------------------------------------------------------
            # INNER LOOP: Evaluate Dataset Samples
            # ---------------------------------------------------------
            for idx, sample in enumerate(tqdm(dl, desc=f"Evaluating {explainer_name}")):
                if args.max_samples is not None and idx >= args.max_samples:
                    break

                if idx in processed_indices:
                    continue

                img = sample["image"]
                # question = sample["question"]
                text = "Write a one sentence caption"
                masks = sample.get("category_masks", {})

                try:
                    # 1. Forward Pass
                    inputs = model_wrapper.get_inputs(img, text)
                    pred_results = model_wrapper.predict(inputs, return_logits=True)

                    # 2. Tokenize and Group Words
                    tokens = pred_results["new_ids"].cpu().unsqueeze(0).tolist()
                    if isinstance(tokens, int):  # Edge case: model generated exactly 1 token
                        tokens = [tokens]

                    words, tokens_id_groups = ids_to_word_groups(tokens, model_wrapper.processor)

                    # 3. Pre-Filter to find Valid Object Targets
                    valid_words, target_indices = plaus_metrics.get_valid_targets(words, tokens_id_groups, masks)

                    # Skip if no objects were generated
                    if not target_indices:
                        log_dict = {
                            "sample_idx": idx,
                            "image_id": sample.get("image_id", f"unknown_{idx}"),
                            "explainer": explainer_name,
                            "prediction": pred_results.get("text"),
                            "note": "No valid objects generated.",
                        }
                        # wandb.log(log_dict, step=idx)
                        save_to_jsonl(log_dict, output_file)
                        torch.cuda.empty_cache()
                        continue

                    # 4. Generate Attributions (ONLY for the targeted nouns)
                    start_time = time.perf_counter()
                    _, img_attrs = explainer.attribute(
                        img,
                        text=text,
                        target_indices=target_indices,
                        pred_results=pred_results,
                    )
                    text_attrs = None
                    xai_gen_time = time.perf_counter() - start_time

                    # 5. Package XAI Result
                    # BUG FIX: Because the explainer ONLY generated heatmaps for target_indices,
                    # we do NOT slice [yes_no_tok_idx] here. We pass the whole returned tensor.
                    xai_result = {
                        "inputs": inputs,
                        "target_ids": pred_results["new_ids"].unsqueeze(0),
                        "pixel_attribution": img_attrs,
                        "token_attribution": text_attrs,
                        "valid_words": valid_words,  # Crucial for Plausibility to map rows to words
                    }

                    faith_sample = {"image": img, "text": text}

                    # 6. Compute Both Metrics Using the Exact Same Data!
                    faith_scores = faith_metrics.compute(model_wrapper, faith_sample, xai_result)
                    plaus_scores = plaus_metrics.compute(model_wrapper, sample, xai_result)

                    # 7. Logging
                    log_dict = {
                        "sample_idx": idx,
                        "image_id": sample.get("image_id", f"unknown_{idx}"),
                        "explainer": explainer_name,
                        "text": text,
                        "label": sample.get("label"),
                        "prediction": pred_results.get("text"),
                        "xai_gen_time": xai_gen_time,
                    }
                    log_dict.update(faith_scores)
                    log_dict.update(plaus_scores)

                    # wandb.log(log_dict, step=idx)
                    save_to_jsonl(log_dict, output_file)

                    del pred_results, img_attrs, text_attrs
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[!] Explainer failed on sample {idx}: {e}")
                    continue

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
            # Cleanup
            print(f"[*] Finished {explainer_path}. Cleaning up GPU memory...")
            # 1. Safely force-delete the explainer from VRAM if it initialized
            if explainer is not None:
                print("[*] Cleaning up GPU memory...")
                if hasattr(explainer, "cleanup"):
                    with contextlib.suppress(Exception):
                        explainer.cleanup()

                del explainer
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM XAI Benchmark")

    # CLI Arguments now just point to YAML files!
    parser.add_argument("--model_config", type=str, required=True, help="Path to model YAML")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument(
        "--explainers",
        nargs="+",
        required=True,
        help="List of paths to explainer YAMLs",
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU node to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/test_caption",
        help="Where to save logs",
    )

    args = parser.parse_args()

    try:
        run_benchmark(args)
    except Exception as e:
        print(f"[FATAL] Benchmark crashed: {e}")
        # wandb.finish(exit_code=1)
