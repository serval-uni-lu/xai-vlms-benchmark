import argparse
import contextlib
import os
import time
import traceback

import torch
from tqdm import tqdm

# Ensure you import the new function we built!
from src.datasets.factory import get_dataloader
from src.explainers.factory import get_explainer
from src.metrics import FaithfulnessMetric
from src.models.factory import load_vlm
from src.utils.xai_utils import find_mcvqa_token_index, get_processed_indices, load_yaml, save_to_jsonl


def run_benchmark(args):
    # 1. Load Configurations
    dataset_config = load_yaml(args.dataset_config)
    model_config = load_yaml(args.model_config)

    explainer_paths = []
    for exp in args.explainers:
        if exp.endswith(".yaml"):
            explainer_paths.append(exp)
        else:
            explainer_paths.append(f"configs/explainers/{exp}.yaml")

    explainer_configs = [load_yaml(path) for path in explainer_paths]

    output_dir = os.path.join(args.output_dir, model_config["name"], f"{dataset_config['name']}")
    os.makedirs(output_dir, exist_ok=True)

    needs_attention = any(cfg.get("requires_attention", False) for cfg in explainer_configs)
    attn_mode = "eager" if needs_attention else None
    print(f"[*] Attention Implementation set to: {attn_mode}")

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

    dl = get_dataloader(dataset_config)

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

    for explainer_path in explainer_paths:
        explainer = None
        try:
            explainer, explainer_name = get_explainer(explainer_path, model_wrapper, model_config)

            print(
                f"\n{'=' * 50}\n[*] Evaluating: {explainer_name} on \
                  {model_config['name']}\n{'=' * 50}"
            )

            run_name = f"{model_config['name']}_{dataset_config['name']}_{explainer_name}"
            output_file = os.path.join(output_dir, f"{run_name}_results.jsonl")

            # --- RESUME LOGIC ---
            processed_indices = get_processed_indices(
                output_file=output_file, total_dataset_len=len(dl), max_samples=args.max_samples
            )

            # --- RATIO TRACKERS ---
            total_processed_this_run = 0
            parse_failures_this_run = 0

            # ---------------------------------------------------------
            # INNER LOOP: Evaluate Dataset Samples
            # ---------------------------------------------------------
            for idx, sample in enumerate(tqdm(dl, desc=f"Evaluating {explainer_name}")):
                if args.max_samples is not None and idx >= args.max_samples:
                    break

                if idx in processed_indices:
                    continue
                if model_config["name"] == "llava":
                    constraint = "\nAnswer"
                else:
                    constraint = "\nAnswer directly with only the letter inside parentheses,\
                        and nothing else. \nAnswer:"
                img = sample["image"]
                question = sample["question"] + constraint
                label = sample["label"]
                category = sample["category"]

                try:
                    # 1. Forward Pass
                    inputs = model_wrapper.get_inputs(img, question)
                    pred_results = model_wrapper.predict(inputs, return_logits=True)

                    full_text = pred_results.get("text", "")
                    if isinstance(full_text, list):
                        full_text = full_text[0]

                    # 2. Identify the Decision Token using our new function
                    # It returns a tuple: (index, token_string)
                    decision_idx, decision_token_str = find_mcvqa_token_index(pred_results["new_ids"], tokenizer=tok)

                    total_processed_this_run += 1

                    # --- FAILURE HANDLING: If the parser couldn't find the answer ---
                    if decision_idx == -1:
                        parse_failures_this_run += 1

                        # Log the failure so you can analyze the bad text later, but SKIP the XAI math
                        # log_dict = {
                        #     "sample_idx": idx,
                        #     "explainer": explainer_name,
                        #     "question": question,
                        #     "label": label,
                        #     "category": category,
                        #     "prediction": full_text,
                        #     "parse_success": False,
                        #     "decision_token_idx": -1,
                        #     "decision_token_str": None,
                        # }
                        # save_to_jsonl(log_dict, output_file)
                        torch.cuda.empty_cache()
                        continue

                    # --- SUCCESS HANDLING: Run the Explainer ---
                    start_time = time.perf_counter()
                    text_attrs, img_attrs = explainer.attribute(
                        img,
                        text=question,
                        target_indices=[decision_idx],  # Using the specific index!
                        pred_results=pred_results,
                    )
                    xai_gen_time = time.perf_counter() - start_time

                    xai_result = {
                        "inputs": inputs,
                        "target_ids": pred_results["new_ids"].unsqueeze(0),
                        "pixel_attribution": img_attrs[0:1],
                        "token_attribution": text_attrs[0:1],
                    }

                    faith_sample = {"image": img, "text": question}
                    faith_scores = faith_metrics.compute(model_wrapper, faith_sample, xai_result)

                    # 5. Full Logging with Token Info
                    log_dict = {
                        "sample_idx": idx,
                        "explainer": explainer_name,
                        "question": question,
                        "label": label,
                        "category": category,
                        "prediction": full_text,
                        "parse_success": True,
                        "decision_token_idx": decision_idx,  # Logged for analysis
                        "decision_token_str": decision_token_str,  # Logged for analysis
                        "xai_gen_time": xai_gen_time,
                    }
                    log_dict.update(faith_scores)
                    save_to_jsonl(log_dict, output_file)

                    del pred_results, img_attrs, text_attrs
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Explainer {explainer_name} failed on sample {idx}: {e}")
                    traceback.print_exc()
                    continue

            # --- PRINT FINAL RATIOS ---
            if total_processed_this_run > 0:
                failure_rate = (parse_failures_this_run / total_processed_this_run) * 100
                print(f"\n[*] Finished {explainer_name}.")
                print(
                    f"[*] Parse Diagnostics: {parse_failures_this_run} failed out of \
                      {total_processed_this_run} attempted ({failure_rate:.2f}% failure rate)."
                )

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
    parser = argparse.ArgumentParser(description="Run VLM XAI Benchmark")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--explainers", nargs="+", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="logs/results_test_baseline")

    args = parser.parse_args()
    try:
        run_benchmark(args)
    except Exception as e:
        print(f"[FATAL] Benchmark crashed: {e}")
