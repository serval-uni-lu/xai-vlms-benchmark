import glob
import itertools
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import wilcoxon

SAVE_DIR = "../figs"


def load_all_benchmark_data(
    base_dir="logs/results",
    # dataset_name="repope_all"
):
    """
    Crawls the directory structure to load all JSONL files,
    automatically extracting the Model and Dataset from folder names.
    """
    all_data = []
    # Adjust this glob pattern based on your actual folder structure!
    # e.g., logs/llava/pope/ig_results.jsonl
    search_pattern = os.path.join(base_dir, "*", "*", "*.jsonl")
    file_list = glob.glob(search_pattern)

    print(f"[*] Found {len(file_list)} result files. Aggregating...")

    for file_path in file_list:
        # Extract metadata from the path
        # (e.g., parts[-3] = model, parts[-2] = dataset)
        path_parts = os.path.normpath(file_path).split(os.sep)
        if len(path_parts) >= 3:
            model_name = path_parts[-3]
            dataset_name = path_parts[-2]
        else:
            model_name, dataset_name = "Unknown", "Unknown"

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        # Ensure we don't overwrite if the JSON already has it
                        row["Model"] = row.get("model", model_name)
                        row["Dataset"] = row.get("dataset", dataset_name)
                        # Ensure explainer name is clean
                        row["Explainer"] = row.get("explainer", os.path.basename(file_path).replace(".jsonl", ""))
                        all_data.append(row)
                    except json.JSONDecodeError:
                        continue

    df = pd.DataFrame(all_data)
    # df['Explainer'].replace('LXT', 'AttnLRP', inplace=True)
    df["Explainer"] = df["Explainer"].replace("LXT", "AttnLRP")

    df["tok_srg_norm"] = df["tok_norm_auc_ins"] - df["tok_norm_auc_del"]
    df["tok_srg"] = df["tok_auc_ins"] - df["tok_auc_del"]
    df["img_srg_norm"] = df["img_norm_auc_ins"] - df["img_norm_auc_del"]
    df["img_srg"] = df["img_auc_ins"] - df["img_auc_del"]

    # Optional: Calculate combined AUPC metrics if you saved curves instead of raw AUCs
    # ... (Insert your trapezoid np.trapezoid code here if needed) ...

    return df


def find_motivating_examples(
    df, explainer_name="AttnLRP", dataset_filter=None, auc_del_col="img_norm_auc_del", syn_del_col="syn_synergy_auc"
):
    """
    Mines the benchmark dataframe for perfect OR-gate and AND-gate examples.
    Assumes df has columns: 'sample_idx', 'Explainer', 'Prompt',
    'auc_del_img' (unimodal), and 'auc_del_joint' (multimodal).
    """
    print(f"--- Mining Candidates for {explainer_name} ---")

    # Filter to a specific high-performing explainer so the heatmaps look good
    df_exp = df[df["Explainer"] == explainer_name].copy()

    # Apply the dataset filter if one is provided
    if dataset_filter:
        print(f"[*] Restricting search to Dataset: {dataset_filter}")
        df_exp = df_exp[df_exp["Dataset"] == dataset_filter]

    # ---------------------------------------------------------
    # 1. Find the OR-Gate (Redundant / Language Prior)
    # Confidence stays flat when image is deleted (img_norm_auc_del is HIGH)
    # But confidence crashes when both are deleted (auc_del_joint is LOW)
    # ---------------------------------------------------------
    or_gates = df_exp[(df_exp[auc_del_col] > 0.65) & (df_exp[syn_del_col] < -0.10)].sort_values(
        auc_del_col, ascending=False
    )

    print(f"\n[*] Found {len(or_gates)} perfect OR-Gate candidates.")
    print("Top 3 OR-Gate Sample IDs:")
    print(or_gates[["sample_idx", "question", auc_del_col, "label", syn_del_col, "Model"]].head(3))

    # ---------------------------------------------------------
    # 2. Find the AND-Gate (Strict Visual Grounding)
    # Confidence crashes immediately when image is deleted auc_del_col is LOW)
    # ---------------------------------------------------------
    and_gates = df_exp[(df_exp[auc_del_col] < 0.15) & (df_exp[syn_del_col] > 0.20)].sort_values(
        auc_del_col, ascending=True
    )

    print(f"\n[*] Found {len(and_gates)} perfect AND-Gate candidates.")
    print("Top 3 AND-Gate Sample IDs:")
    print(and_gates[["sample_idx", "question", auc_del_col, "label", "Model"]].head(3))

    return or_gates, and_gates


def mine_qualitative_archetypes(df):
    print("--- Mining Qualitative Examples for the Appendix ---")

    # Exact column names from your dataframe
    f_syn_col = "syn_synergy_norm_auc"
    mui_col = "img_srg_norm"
    mut_col = "tok_srg_norm"

    # 1. Pivot the dataframe using pivot_table (Bulletproof against duplicates)
    # By including 'Model' in the index, we separate Qwen, LLaVA, and InternVL!
    pivot_df = pd.pivot_table(
        df,
        index=["sample_idx", "Dataset", "Model", "question", "label"],
        columns="Explainer",
        values=[f_syn_col, mui_col, mut_col],
        aggfunc="first",  # Grabs the first value if there are any weird duplicate rows
    ).reset_index()

    # Flatten the multi-index columns (e.g., 'img_srg_norm' + 'TAM' -> 'img_srg_norm_TAM')
    pivot_df.columns = ["_".join(col).strip("_") for col in pivot_df.columns.values]

    # ==========================================
    # Archetype 1: The Visual Salience Trap
    # ==========================================
    # TAM highlights the object visually (High mu_I) but fails text synergy (Low F_syn).
    # AttnLRP succeeds at the synergy (High F_syn).
    trap_mask = (
        (pivot_df[f"{mui_col}_TAM"] > 0.40)  # Adjust thresholds based on your data distribution
        & (pivot_df[f"{f_syn_col}_TAM"] < 0.10)
        & (pivot_df[f"{f_syn_col}_Rollout"] > 0.10)  # Assuming AttnLRP is in your dataset!
    )
    salience_traps = pivot_df[trap_mask]
    print(f"\n[*] Found {len(salience_traps)} 'Visual Salience Trap' candidates.")
    if not salience_traps.empty:
        print("Top Candidate:")
        print(salience_traps[["sample_idx", "Dataset", "Model", "question"]].head(1))

    # ==========================================
    # Archetype 2: Unimodal Contradiction (Ranking Instability)
    # ==========================================
    # An explainer looks perfect visually but completely ignores the text.
    contradiction_mask = (
        (pivot_df[f"{mui_col}_InputxGradients"] > 0.50)  # Looks great visually
        & (pivot_df[f"{mut_col}_InputxGradients"] < -0.10)  # But text score is collapsed
    )
    contradictions = pivot_df[contradiction_mask]
    print(f"\n[*] Found {len(contradictions)} 'Unimodal Contradiction' candidates.")
    if not contradictions.empty:
        print("Top Candidate:")
        print(contradictions[["sample_idx", "Dataset", "Model", "question"]].head(1))

    # ==========================================
    # Archetype 3: The Universal Failure
    # ==========================================
    # A brutally hard MMStar question where NO explainer gets a good synergy score.
    failure_mask = (
        (pivot_df["Dataset"] == "mmstar")
        & (pivot_df[f"{f_syn_col}_Rollout"] < 0.05)
        & (pivot_df[f"{f_syn_col}_TAM"] < 0.05)
    )
    failures = pivot_df[failure_mask]
    print(f"\n[*] Found {len(failures)} 'Universal Failure' candidates.")
    if not failures.empty:
        print("Top Candidate:")
        print(failures[["sample_idx", "Dataset", "Model", "question"]].head(1))

    return salience_traps, contradictions, failures


def plot_unimodal_deletion_curve(
    perturbations, scores, auc_val=None, curve_type="OR-gate", savefig=False, format="png", SAVE_DIR="../figs"
):
    """
    Plots a highly polished, NeurIPS-ready Unimodal Deletion Curve.
    Dynamically colors the plot based on whether it is an OR-gate (Failure) or AND-gate (Success).
    """
    # 1. Plotting Setup
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)  # Slightly square-ish for side-by-side panels

    # 2. Dynamic Styling (Red/Failure for OR-gate, Blue/Success for AND-gate)
    if curve_type == "OR-gate":
        line_color = "#e74c3c"  # Red
        title = "OR-Gate example (Modality bias)"
    else:
        line_color = "#2980b9"  # Blue
        title = "AND-Gate example (Visual grounding)"

    # 3. Plot the Line and the Area Under the Curve (AUC)
    ax.plot(
        perturbations,
        scores,
        color=line_color,
        linewidth=3,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        label=r"$del_{img}(k)$",
    )

    # This is the crucial visual upgrade: shading the AUC area
    ax.fill_between(perturbations, scores, color=line_color, alpha=0.15)

    # 4. Add the exact AUC value in a clean bounding box
    if auc_val is not None:
        props = {"boxstyle": "round", "pad": 0.4, "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"}
        # Place it in the bottom left for OR-gate (where there is empty space)
        # and top right for AND-gate
        y_pos = 0.15 if curve_type == "OR-gate" else 0.85
        ax.text(
            0.05,
            y_pos,
            f"AUC = {auc_val:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=line_color,
            bbox=props,
        )

    # 5. Strict Formatting
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)  # Cap slightly above 1.0 for visual breathing room

    ax.set_xlabel(r"Fraction of Pixels Masked ($k$)", fontweight="bold", labelpad=10)
    ax.set_ylabel("VLM Confidence Score", fontweight="bold", labelpad=10)
    ax.set_title(title, fontweight="bold", pad=15, fontsize=13)

    # Clean up grid and spines
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # 6. Save and Show
    if savefig:
        filename = f"deletion_curve_{curve_type.lower().replace('-', '_')}.{format}"
        save_path = os.path.join(SAVE_DIR, filename)
        plt.savefig(save_path, format=format, bbox_inches="tight")
        print(f"[*] Saved figure to {save_path}")

    plt.show()


# ==========================================
# Correlation plots
# ==========================================


def load_and_preprocess_data(results_file):
    """Loads JSONL data and computes the necessary AUC columns."""
    data = []
    with open(results_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))

    df = pd.DataFrame(data)

    pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Compute AUCs if they don't exist
    for col in ["syn_del", "syn_ins", "syn_del_norm", "syn_ins_norm"]:
        curve_col = f"{col}_synergy_curve"
        auc_col = f"{col}_auc"
        if auc_col not in df.columns and curve_col in df.columns:
            df[auc_col] = df[curve_col].apply(lambda val: np.trapezoid(val, x=pert_steps))

    # Calculate the unnormalized synergy
    if "syn_ins_auc" in df.columns and "syn_del_auc" in df.columns:
        df["synergy_srg_unorm"] = (df["syn_ins_auc"] + df["syn_del_auc"]) / 2.0

    return df


def load_all_correlation_data(
    base_dir="logs/correlation",
    # dataset_name="repope_all"
):
    """
    Crawls the directory structure to load all JSONL files,
    automatically extracting the Model and Dataset from folder names.
    """
    all_data = []
    # Adjust this glob pattern based on your actual folder structure!
    # e.g., logs/llava/pope/ig_results.jsonl
    search_pattern = os.path.join(base_dir, "*", "*", "*.jsonl")
    file_list = glob.glob(search_pattern)

    print(f"[*] Found {len(file_list)} result files. Aggregating...")

    for file_path in file_list:
        # Extract metadata from the path
        # (e.g., parts[-3] = model, parts[-2] = dataset)
        path_parts = os.path.normpath(file_path).split(os.sep)
        if len(path_parts) >= 3:
            model_name = path_parts[-3]
            dataset_name = path_parts[-2]
        else:
            model_name, dataset_name = "Unknown", "Unknown"

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        # Ensure we don't overwrite if the JSON already has it
                        row["Model"] = row.get("model", model_name)
                        row["Dataset"] = row.get("dataset", dataset_name)
                        # Ensure explainer name is clean
                        row["Explainer"] = row.get(
                            "explainer", os.path.basename(file_path).replace(".jsonl", "").split("_")[3]
                        )
                        all_data.append(row)
                    except json.JSONDecodeError:
                        continue

    df = pd.DataFrame(all_data)
    df["Explainer"] = df["Explainer"].replace("Lxt", "AttnLRP")
    pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    cols_to_drop = ["final_alignment", "final_true_syn", "synergy_refined_srg", "final_redundancy", "standard_srg"]
    df = df.drop(cols_to_drop, axis=1)

    # Compute AUCs if they don't exist
    for col in ["syn_del", "syn_ins", "syn_del_norm", "syn_ins_norm"]:
        curve_col = f"{col}_synergy_curve"
        auc_col = f"{col}_auc"
        if auc_col not in df.columns and curve_col in df.columns:
            df[auc_col] = df[curve_col].apply(lambda val: np.trapezoid(val, x=pert_steps))

    # Calculate the unnormalized synergy
    if "syn_ins_auc" in df.columns and "syn_del_auc" in df.columns:
        df["synergy_srg_unorm"] = (df["syn_ins_auc"] + df["syn_del_auc"]) / 2.0

    # Optional: Calculate combined AUPC metrics if you saved curves instead of raw AUCs
    # ... (Insert your trapezoid np.trapezoid code here if needed) ...

    return df


def analyze_and_plot_correlation(df, x_col, y_col, x_label=None, y_label=None, title=None, save_path=None):
    """
    Cleans data, computes correlation statistics, prints a report, and generates a paper-ready plot.
    """
    # 1. Clean Data: Drop rows where either metric is NaN
    clean_df = df.dropna(subset=[x_col, y_col]).copy()

    if len(clean_df) == 0:
        print(f"[!] Cannot analyze {x_col} vs {y_col}: No valid overlapping data points.")
        return

    # 2. Compute Correlations
    pearson_r, p_val_p = stats.pearsonr(clean_df[x_col], clean_df[y_col])
    spearman_rho, p_val_s = stats.spearmanr(clean_df[x_col], clean_df[y_col])

    # 3. Print Report
    print("\n" + "=" * 50)
    print(f"      XAI METRIC VALIDATION: {x_col} vs {y_col}")
    print("=" * 50)
    print(f"Total Samples Analyzed : {len(clean_df)}")
    print(f"Metric X ({x_col}) Mean : {clean_df[x_col].mean():.4f} ± {clean_df[x_col].std():.4f}")
    print(f"Metric Y ({y_col}) Mean : {clean_df[y_col].mean():.4f} ± {clean_df[y_col].std():.4f}")
    print("\n--- Correlation ---")
    print(f"Pearson r      : {pearson_r:.4f}  (p-value: {p_val_p:.2e})")
    print(f"Spearman rho   : {spearman_rho:.4f}  (p-value: {p_val_s:.2e})")
    print("=" * 50)

    # 4. Generate Publication-Ready Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))

    ax = sns.regplot(
        x=x_col,
        y=y_col,
        data=clean_df,
        scatter_kws={"alpha": 0.6, "color": "#2C3E50"},
        line_kws={"color": "#E74C3C", "linewidth": 2},
    )

    # Use provided labels or default to column names
    plt.title(title or f"Validation: {x_col} vs {y_col}", fontsize=14, weight="bold")
    plt.xlabel(x_label or x_col, fontsize=12)
    plt.ylabel(y_label or y_col, fontsize=12)

    # Add Stats Box
    textstr = f"Spearman $\\rho$ = {spearman_rho:.3f}\nPearson $r$ = {pearson_r:.3f}"
    props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"}
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[*] Plot saved to {save_path}")

    plt.show()  # Display the plot
    plt.close()


def plot_correlation_scatter(
    df, x_col="sii_auc", y_col="synergy_srg", savefig=False, format="pdf", save_path="correlation_scatter"
):

    print("--- Generating 2x2 Correlation Subplots (Hued by Model) ---")

    # 1. Drop NaNs and extract necessary columns
    plot_df = df[[x_col, y_col, "Explainer", "Model"]].dropna()

    # Specify the 4 explainers you want to plot
    target_explainers = ["Random", "InputxGradients", "Rollout", "TAM"]
    plot_df = plot_df[plot_df["Explainer"].isin(target_explainers)]

    # --- NEW: Calculate Global Correlation across ALL included points ---
    global_spearman, _ = stats.spearmanr(plot_df[x_col], plot_df[y_col])
    global_pearson, _ = stats.pearsonr(plot_df[x_col], plot_df[y_col])

    # 2. Setup a 2x2 grid sharing both axes
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300, sharex=True, sharey=True)

    # Define a clean color palette for the models (adjust n_colors if you have more than 3 models)
    model_palette = sns.color_palette("Set1", n_colors=plot_df["Model"].nunique())

    for ax, explainer in zip(axes.flat, target_explainers, strict=False):
        sub_df = plot_df[plot_df["Explainer"] == explainer]

        # 3. Scatter Plot with HUE per Model
        sns.scatterplot(
            data=sub_df,
            x=x_col,
            y=y_col,
            hue="Model",
            palette=model_palette,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            s=20,
            ax=ax,
            legend=False,  # We will create one global legend later!
        )

        # 4. Global Line of Best Fit for this explainer (regardless of model)
        if len(sub_df) > 1:
            m, b = np.polyfit(sub_df[x_col], sub_df[y_col], 1)
            x_range = np.linspace(sub_df[x_col].min(), sub_df[x_col].max(), 100)
            ax.plot(x_range, m * x_range + b, color="black", linestyle="--", linewidth=2, alpha=0.7)

            # Calculate Correlations
            spearman_rho, _ = stats.spearmanr(sub_df[x_col], sub_df[y_col])
            pearson_r, _ = stats.pearsonr(sub_df[x_col], sub_df[y_col])

            # Add the text box inside the subplot
            stats_text = f"$\\rho$ = {spearman_rho:.2f}\n$r$ = {pearson_r:.2f}"
            props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"}
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=props,
                fontweight="bold",
            )

        # 5. Subplot Formatting
        ax.set_title(f"Explainer: {explainer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")  # Clear individual labels, we will use global ones
        ax.set_ylabel("")
        ax.grid(True, linestyle=":", alpha=0.6)

    # 6. Global Labels and Legend
    fig.supxlabel("Exact Shapley Interaction Index (SII)", fontsize=14, fontweight="bold")
    fig.supylabel(r"Proposed Synergistic Faithfulness ($\mathcal{F}_{syn}$)", fontsize=14, fontweight="bold")

    # --- NEW: Add the Global Title ---
    global_title = f"Global Correlation with Exact Shapley ($\\rho$ = {global_spearman:.2f};\
        $\\tau$ = {global_pearson:.2f})"
    fig.suptitle(global_title, fontsize=14, fontweight="bold", y=1.06)

    # Extract legend handles from a dummy plot to create a single global legend
    handles, labels = ax.get_legend_handles_labels()
    # If the standard handle extraction misses, create a dummy legend:
    legend_elements = [
        mpatches.Patch(color=color, label=model)
        for color, model in zip(model_palette, plot_df["Model"].unique(), strict=False)
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=plot_df["Model"].nunique(),
        title="VLM",
        title_fontsize="12",
        fontsize="11",
        frameon=False,
    )

    plt.tight_layout()
    # Adjust layout so the titles and labels don't overlap
    fig.subplots_adjust(bottom=0.08, left=0.08, top=0.90)

    if savefig:
        save_path = os.path.join(SAVE_DIR, f"{save_path}.{format}")
        plt.savefig(save_path, format=format, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.show()


def plot_runtime_complexity(df, time_col="time_syn", savefig=False, format="png", save_path="runtime_complexity"):
    print("--- Generating 1x2 Runtime Complexity Figure ---")

    # Setup a 1x2 grid
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # ==========================================
    # LEFT PANEL: Theoretical Forward Passes
    # ==========================================
    ax1 = axes[0]

    # Generate theoretical data (Sequence length vs Passes)
    # We stop at N=25 because 2^25 is already 33 million passes!
    tokens = np.arange(2, 25)
    exact_sii_passes = 2**tokens

    # Your metric: 6K + 2. Assuming K=10, that's 62 passes.
    K = 10
    fsyn_passes = np.full_like(tokens, 6 * K + 2)

    ax1.plot(tokens, exact_sii_passes, color="#e74c3c", linewidth=2.5, label=r"Exact SII $\mathcal{O}(2^{N})$")
    ax1.plot(tokens, fsyn_passes, color="#2ecc71", linewidth=2.5, label=r"Proposed $\mathcal{F}_{{syn}}$ ($K={K}$)")

    # Use a Log Scale for the Y-axis because Exact SII explodes instantly
    ax1.set_yscale("log")
    ax1.set_xlabel("Total Sequence Length ($m + n$ tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Forward Passes (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Theoretical Complexity", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # ==========================================
    # RIGHT PANEL: Empirical Wall-Clock Time
    # ==========================================
    ax2 = axes[1]

    # Sort explainers by mean time so the bar chart looks clean
    mean_times = df.groupby("Explainer")[time_col].mean().sort_values()
    order = mean_times.index.tolist()

    # Create the barplot. Seaborn automatically calculates the 95% CI or std for error bars!
    sns.barplot(
        data=df,
        x="Explainer",
        y=time_col,
        order=order,
        color="#3498db",
        edgecolor="black",
        err_kws={"linewidth": 1.5, "color": "black"},
        capsize=0.1,
        ax=ax2,
    )

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_xlabel("Explainer Method", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Average Evaluation Time (seconds)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Empirical Wall-Clock Time per Instance", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Clean up spines
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if savefig:
        save_path = os.path.join(SAVE_DIR, f"{save_path}.{format}")
        plt.savefig(save_path, format=format, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.show()


def plot_time_comparison(
    df, sii_col="time_sii", syn_col="time_syn", savefig=False, format="png", save_path="runtime_comparison_bars"
):
    print("--- Generating Empirical Runtime Comparison Bar Plot ---")

    # 1. Filter the dataset
    # We only want to plot the rows where Exact SII was actually computed (the N=200 subset)
    plot_df = df.dropna(subset=[sii_col, syn_col]).copy()

    # Optional: If you only evaluated certain explainers for the exact SII,
    # we can sort them so the chart looks organized.
    target_explainers = ["Random", "InputxGradients", "Rollout", "TAM"]
    plot_df = plot_df[plot_df["Explainer"].isin(target_explainers)]

    # 2. Reshape the DataFrame using melt()
    # This turns our two time columns into rows so Seaborn can plot them side-by-side
    melted_df = plot_df.melt(
        id_vars=["Explainer"], value_vars=[sii_col, syn_col], var_name="Method", value_name="Time_Seconds"
    )

    # Rename the variables so they look beautiful in the legend
    melted_df["Method"] = melted_df["Method"].replace(
        {sii_col: "Exact SII (Downsampled)", syn_col: r"$\mathcal{F}_{syn}$"}
    )

    # 3. Setup the plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot side-by-side bars with standard deviation error bars
    sns.barplot(
        data=melted_df,
        x="Explainer",
        y="Time_Seconds",
        hue="Method",
        palette=["#e74c3c", "#2ecc71"],  # Red for Exact (bad/slow), Green for Proposed (good/fast)
        edgecolor="black",
        err_kws={"linewidth": 1.5, "color": "black"},
        capsize=0.1,
        ax=ax,
    )

    # 4. Critical Formatting: Logarithmic Scale
    ax.set_yscale("log")

    # 5. Labels and Clean up
    ax.set_xlabel("Explainer method", fontsize=10, fontweight="bold")
    ax.set_ylabel("Execution time per instance (seconds) [Log Scale]", fontsize=10, fontweight="bold")
    ax.set_title(r"Empirical time: Exact SII vs. $\mathcal{F}_{syn}$", fontsize=12, fontweight="bold", pad=15)

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Move the legend outside or to a clean spot so it doesn't cover the bars
    ax.legend(title="Evaluation Method", title_fontsize="11", fontsize="10", loc="best")

    plt.tight_layout()

    if savefig:
        save_path = os.path.join(SAVE_DIR, f"{save_path}.{format}")
        plt.savefig(save_path, format=format, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.show()


# ==========================================
# 2. STATISTICAL TESTING
# ==========================================
def run_statistical_comparison(df, metric="img_ins_auc", baseline_explainer="random", target_explainer="ours"):
    """Runs a Wilcoxon Signed-Rank test between two explainers."""
    print(f"\n--- Statistical Test: {target_explainer} vs {baseline_explainer} ({metric}) ---")

    # We need to compare them on the EXACT same images (Paired Test)
    pivot = df.pivot_table(index=["Model", "Dataset", "sample_idx"], columns="Explainer", values=metric).dropna()

    if baseline_explainer not in pivot.columns or target_explainer not in pivot.columns:
        print("[!] Explainers not found in data for paired testing.")
        return

    baseline_scores = pivot[baseline_explainer]
    target_scores = pivot[target_explainer]

    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(target_scores, baseline_scores)

    mean_diff = target_scores.mean() - baseline_scores.mean()
    print(f"Mean Difference: {mean_diff:+.4f}")
    print(f"p-value        : {p_value:.2e}")
    if p_value < 0.05:
        print("Conclusion     : STATISTICALLY SIGNIFICANT DIFFERENCE (Reject Null)")
    else:
        print("Conclusion     : NO SIGNIFICANT DIFFERENCE")


# ==========================================
# 3. PUBLICATION PLOTS
# ==========================================


# ==========================================
# 1. MACRO BOXPLOT (Variance Distribution)
# ==========================================
def plot_macro_boxplot(df, metric="img_ins_auc", dataset_filter=None, savefig=False, format="pdf"):
    """Generates a grouped boxplot. Filters by dataset if provided."""
    plot_df = df.copy()
    title_suffix = " (All Datasets)"

    if dataset_filter:
        plot_df = plot_df[plot_df["Dataset"] == dataset_filter]
        title_suffix = f" (Dataset: {dataset_filter})"

    if plot_df.empty:
        print(f"[!] No data found for dataset: {dataset_filter}")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(12, 6))

    ranking = plot_df.groupby("Explainer")[metric].median().sort_values(ascending=False).index

    _ = sns.boxplot(x="Explainer", y=metric, hue="Model", data=plot_df, order=ranking, palette="Set2", showfliers=False)

    plt.title(f"Faithfulness score distribution across explainers{title_suffix}", weight="bold")
    plt.xlabel("XAI Method", weight="bold")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Vision-Language Model", loc="best")

    filename = f"boxplot_{metric}_{dataset_filter or 'global'}.{format}"
    plt.tight_layout()
    if savefig:
        filename = os.path.join(SAVE_DIR, filename)
        plt.savefig(filename, dpi=300, format=format)
        print(f"[*] Saved {filename}")
        # plt.close()
    plt.show()


# ==========================================
# 2. PARETO PLOT (Speed vs. Performance)
# ==========================================
def plot_speed_vs_performance(
    df, perf_metric="img_ins_auc", time_metric="xai_gen_time", dataset_filter=None, format="pdf", savefig=False
):
    """Scatter plot showing computation time vs accuracy."""
    plot_df = df.copy()
    title_suffix = " (All Datasets)"

    if dataset_filter:
        plot_df = plot_df[plot_df["Dataset"] == dataset_filter]
        title_suffix = f"\n(Dataset: {dataset_filter})"

    if plot_df.empty or time_metric not in plot_df.columns:
        print(f"[!] Missing data or time metrics for {dataset_filter or 'global'}")
        return

    agg_df = plot_df.groupby("Explainer").agg({perf_metric: "mean", time_metric: "mean"}).reset_index()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))

    ax = sns.scatterplot(x=time_metric, y=perf_metric, data=agg_df, s=150, color="#ED481B", edgecolor="black")

    for line in range(0, agg_df.shape[0]):
        ax.text(
            agg_df[time_metric][line] * 1.05,
            agg_df[perf_metric][line],
            agg_df["Explainer"][line],
            horizontalalignment="left",
            size="small",
            color="black",
            weight="semibold",
        )

    plt.xscale("log")
    plt.title(f"Trade-off: Cost vs. Faithfulness{title_suffix}", weight="bold")
    plt.xlabel("Average Time per Image (Seconds, Log Scale)", weight="bold")
    plt.ylabel(r"Mean $\mathcal{F}_{syn}$", weight="bold")

    plt.axvline(agg_df[time_metric].median(), color="gray", linestyle="--", alpha=0.5)
    plt.axhline(agg_df[perf_metric].median(), color="gray", linestyle="--", alpha=0.5)

    filename = f"pareto_{perf_metric}_{dataset_filter or 'global'}.{format}"
    plt.tight_layout()

    if savefig:
        filename = os.path.join(SAVE_DIR, filename)
        plt.savefig(filename, dpi=300, format=format)
        print(f"[*] Saved {filename}")
        # plt.close()
    plt.show()


# ==========================================
# 3. MEAN RANK & SIGNIFICANCE (Dynamic Winner Anchor)
# ==========================================
def compute_and_plot_ranks(
    df, metric="img_ins_auc", ascending_metric=False, dataset_filter=None, savefig=False, format="pdf"
):
    """
    Computes the Mean Rank per image and plots a sleek bar chart.
    Statistical significance (symbols only) is cleanly tabulated on the right margin,
    with a legend explaining the symbols.
    """
    print("--- Generating Final Mean Rank Plot (Symbols Only) ---")
    plot_df = df.copy()
    title_suffix = "All Datasets"

    if dataset_filter:
        plot_df = plot_df[plot_df["Dataset"] == dataset_filter]
        title_suffix = f"Dataset: {dataset_filter}"

    pivot = plot_df.pivot_table(index=["Model", "Dataset", "sample_idx"], columns="Explainer", values=metric).dropna()

    if pivot.empty:
        print("[!] No valid paired data found. Skipping.")
        return

    # 1. Ranks and Mean Ranks
    ranks_df = pivot.rank(axis=1, ascending=ascending_metric)
    mean_ranks = ranks_df.mean().sort_values()
    k = len(mean_ranks)
    best_explainer = mean_ranks.index[0]

    # 2. Statistical Testing (Wilcoxon vs Winner with Bonferroni correction)
    stats_results = {}
    num_comparisons = k - 1

    for exp in mean_ranks.index:
        if exp == best_explainer:
            stats_results[exp] = {"p_adj": 1.0, "sig": "-"}
            continue

        _, p_raw = wilcoxon(pivot[best_explainer], pivot[exp])
        p_adj = min(1.0, p_raw * num_comparisons)

        if p_adj < 0.001:
            sig_text = "***"
        elif p_adj < 0.01:
            sig_text = "**"
        elif p_adj < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"

        stats_results[exp] = {"p_adj": p_adj, "sig": sig_text}

    # 3. Plotting Setup
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    y_pos = np.arange(k)
    # uniform_color = "#DC610F"

    # Draw horizontal bars
    # bars = ax.barh(y_pos, mean_ranks.values, color=uniform_color,
    #                height=0.6, edgecolor='black', linewidth=1.2, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(mean_ranks.index, weight="bold")
    ax.invert_yaxis()  # Rank 1 at the top

    # 4. Text Annotations and Significance Margin
    max_rank = mean_ranks.max()

    # We need less space now, so we shrink the right margin slightly
    ax.set_xlim(0, max_rank * 1.30)

    # Table Header
    ax.text(
        max_rank * 1.20, -0.8, "Sig.\nvs Top", ha="center", va="center", weight="bold", fontsize=11, color="#2c3e50"
    )

    # Subtle separator line
    ax.axvline(max_rank * 1.10, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    for i, (exp, val) in enumerate(mean_ranks.items()):
        # Bar value annotation
        ax.text(val + 0.1, i, f"{val:.2f}", va="center", fontsize=10, weight="bold", color="#2c3e50")

        # Table row annotation (Symbols only)
        if exp == best_explainer:
            p_text = "Control"
            font_w = "bold"
        else:
            sig = stats_results[exp]["sig"]
            p_text = sig
            font_w = "bold" if sig in ["ns", "***", "**", "*"] else "normal"

        # Draw the symbol
        ax.text(
            max_rank * 1.20,
            i,
            p_text,
            ha="center",
            va="center",
            fontsize=13 if p_text in ["***", "**", "*"] else 11,
            weight=font_w,
            color="#2c3e50",
        )

    # 5. Add Legend for Significance Symbols
    legend_text = "*** $p < 0.001$\n ** $p < 0.01$\n  * $p < 0.05$\n ns  not sig."
    props = {"boxstyle": "round", "pad": 0.4, "facecolor": "white", "alpha": 0.95, "edgecolor": "gray"}
    ax.text(
        0.97,
        0.04,
        legend_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    # 6. Formatting
    ax.set_xlabel("Average Rank (Lower is Better)", weight="bold", labelpad=10)
    ax.set_title(f"Mean Rank & Ordinal Significance ({title_suffix})", weight="bold", pad=25, fontsize=14)

    # Remove borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    filename = f"ranks_{metric}_{dataset_filter or 'global'}.{format}"
    plt.tight_layout()
    if savefig:
        filename = os.path.join(SAVE_DIR, filename)
        plt.savefig(filename, dpi=300, bbox_inches="tight", format=format)
        print(f"[*] Saved {filename}")
        # plt.close()
    plt.show()


def global_ranking_and_friedman_test(df, metric="img_ins_auc", savefig=False, format="pdf"):
    """
    Ranks all explainers globally, runs a Friedman test, and generates
    a Pairwise Significance Heatmap with Bonferroni correction.
    """
    print(f"\n{'=' * 50}\n[*] GLOBAL RANKING & STATISTICAL ANALYSIS ({metric})\n{'=' * 50}")

    # 1. Pivot the data so each row is a unique sample, and columns are the explainers
    # We group by Model, Dataset, and Image ID to ensure perfect paired matching.
    pivot = df.pivot_table(
        index=["Model", "Dataset", "sample_idx"], columns="Explainer", values=metric
    ).dropna()  # Drop rows where ANY explainer failed/crashed

    explainers = pivot.columns.tolist()
    k = len(explainers)

    print(f"[*] Valid Paired Samples: {len(pivot)} (Evaluated successfully by ALL {k} explainers)")

    # 2. Print the Global Mean Ranking
    print("\n--- Mean Scores (Global) ---")
    mean_scores = pivot.mean().sort_values(ascending=False)
    for rank, (exp, score) in enumerate(mean_scores.items(), 1):
        print(f"{rank}. {exp:<20} : {score:.4f}")

    # 3. The Friedman Test
    # Unpack the columns into separate arrays for the test
    args = [pivot[col] for col in explainers]
    stat, p_value = stats.friedmanchisquare(*args)

    print("\n--- Step 1: Friedman Test ---")
    print(f"Chi-Square Stat : {stat:.3f}")
    print(f"p-value         : {p_value:.2e}")

    if p_value >= 0.05:
        print("[!] No significant difference found among the explainers. Stopping here.")
        return
    else:
        print("[*] Significant difference detected! Proceeding to Post-Hoc Pairwise tests.")

    # 4. Post-Hoc Pairwise Wilcoxon with Bonferroni Correction
    num_comparisons = (k * (k - 1)) / 2
    print(f"\n--- Step 2: Pairwise Tests (Bonferroni Penalty Multiplier: {num_comparisons}) ---")

    # Create an empty matrix to store the adjusted p-values
    p_matrix = pd.DataFrame(1.0, index=mean_scores.index, columns=mean_scores.index)

    for exp1, exp2 in itertools.combinations(mean_scores.index, 2):
        # Run standard Wilcoxon
        w_stat, p_val_raw = stats.wilcoxon(pivot[exp1], pivot[exp2])

        # Apply Bonferroni Correction (multiply p-value by number of comparisons)
        p_val_adj = min(1.0, p_val_raw * num_comparisons)

        # Store symmetrically in the matrix
        p_matrix.loc[exp1, exp2] = p_val_adj
        p_matrix.loc[exp2, exp1] = p_val_adj

    # 5. Plot the Pairwise Significance Heatmap
    plot_significance_heatmap(p_matrix, metric, savefig=savefig, format=format)


def plot_significance_heatmap(p_matrix, metric, savefig=False, format="pdf"):
    """
    Plots a heatmap showing which explainers are significantly different from each other.
    Green = Significantly Different (p < 0.05)
    Red/Gray = Statistically Tied (p >= 0.05)
    """
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    plt.figure(figsize=(10, 8))

    # Create a binary mask: True if significant, False if not
    # is_significant = p_matrix < 0.05

    # Plotting: We use a custom colormap.
    # Let's show the actual adjusted p-values, but color them by significance.
    # cmap = sns.diverging_palette(10, 130, as_cmap=True)

    # Generate a mask for the upper triangle so we don't duplicate data visually
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))

    _ = sns.heatmap(
        p_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm_r",
        vmin=0,
        vmax=0.1,
        linewidths=0.5,
        cbar_kws={"label": "Adjusted p-value"},
    )

    plt.title(f"Pairwise Statistical Significance ({metric})\n(Bonferroni-Adjusted Wilcoxon, p < 0.05)", weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if savefig:
        savepath = os.path.join(SAVE_DIR, f"pairwise_significance_{metric}.{format}")
        plt.savefig(savepath, dpi=300, format=format)
    plt.show()


def generate_latex_table(df, metric="img_ins_auc"):
    """
    Automatically generates a Pandas Pivot Table and exports the LaTeX code
    for you to copy-paste directly into Overleaf!
    """
    print(f"\n--- Generating LaTeX Table for {metric} ---")
    pivot = df.pivot_table(index="Explainer", columns=["Model", "Dataset"], values=metric, aggfunc="mean")

    # Sort explainers by their overall average across all columns
    pivot["Overall_Mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values(by="Overall_Mean", ascending=False).drop(columns=["Overall_Mean"])

    print(pivot.to_latex(float_format="%.3f"))
    return pivot


# ==========================================
# Metrics limitation plots
# ==========================================

# --- CONFIGURATION ---
IMG_METRIC = "img_srg_norm"
TOK_METRIC = "tok_srg_norm"


def compute_kendalls_tau(df, img_col, tok_col):
    print("=== Kendall's Tau: Image vs Text Ranking Instability ===")

    datasets = df["Dataset"].unique()
    results = []

    for ds in datasets:
        df_ds = df[df["Dataset"] == ds]

        # Get the mean score for each explainer on this dataset
        means = df_ds.groupby("Explainer")[[img_col, tok_col]].mean()

        # Rank them (higher score = rank 1, etc.)
        # Adjust ascending=False/True depending on if higher is better for your metric
        img_ranks = means[img_col].rank(ascending=False)
        tok_ranks = means[tok_col].rank(ascending=False)

        # Compute Kendall's Tau
        tau, p_value = stats.kendalltau(img_ranks, tok_ranks)

        results.append({"Dataset": ds, "Kendalls_Tau": tau, "p_value": p_value})
        print(f"- {ds}: Tau = {tau:.3f} (p={p_value:.3f})")

    # Overall Global Tau
    global_means = df.groupby("Explainer")[[img_col, tok_col]].mean()
    global_tau, global_p = stats.kendalltau(
        global_means[img_col].rank(ascending=False), global_means[tok_col].rank(ascending=False)
    )
    print(f"- GLOBAL (All Datasets): Tau = {global_tau:.3f} (p={global_p:.3f})")
    print("========================================================\n")

    return pd.DataFrame(results)


def plot_rank_shift_bump_chart(
    df,
    img_col,
    tok_col,
    savefig=False,
    format="pdf",
    save_path="rank_shift",
):
    # 1. Calculate global mean scores and rank them
    means = df.groupby("Explainer")[[img_col, tok_col]].mean()

    # rank() assigns lower numbers to lower values by default.
    # We want the best explainer (highest score) to be Rank 1.
    means["Img_Rank"] = means[img_col].rank(ascending=False)
    means["Tok_Rank"] = means[tok_col].rank(ascending=False)

    # Sort by Image Rank for plotting order
    means = means.sort_values("Img_Rank")
    explainers = means.index.tolist()

    # 2. Setup the Plot for NeurIPS (Wide and short to save vertical space)
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

    # Coordinates for the two horizontal axes
    y_img, y_tok = 1, 0

    # 3. Draw the lines and points
    colors = plt.cm.tab10(np.linspace(0, 1, len(explainers)))

    for i, explainer in enumerate(explainers):
        r_img = means.loc[explainer, "Img_Rank"]
        r_tok = means.loc[explainer, "Tok_Rank"]

        # Plot the line connecting the two ranks (X=rank, Y=modality)
        ax.plot(
            [r_img, r_tok], [y_img, y_tok], marker="o", markersize=8, linewidth=2.5, color=colors[i], label=explainer
        )

        # Add Explainer text labels above the top line and below the bottom line.
        # We rotate them 40 degrees so the names don't overlap horizontally.
        ax.text(
            r_img,
            y_img + 0.1,
            f"{explainer} ({int(r_img)})",
            ha="left",
            va="bottom",
            rotation=20,
            fontsize=9,
            fontweight="bold",
            color=colors[i],
        )

        ax.text(
            r_tok,
            y_tok - 0.1,
            f"({int(r_tok)}) {explainer}",
            ha="right",
            va="top",
            rotation=20,
            fontsize=9,
            fontweight="bold",
            color=colors[i],
        )

    # 4. Formatting the axes to look like a clean Slope Graph
    # Set X limits and invert so Rank 1 is on the far left
    ax.set_xlim(len(explainers) + 0.5, 0.5)

    # Set Y limits to give the rotated text plenty of breathing room
    ax.set_ylim(-0.6, 1.6)

    # Remove all borders/spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])  # Remove default x-ticks
    ax.set_yticks([y_tok, y_img])

    # Add the labels to the Y-axis
    ax.set_yticklabels(["Unimodal\nText Metric", "Unimodal\nImage Metric"], fontsize=10, fontweight="bold")

    # Draw horizontal lines to act as the axes
    ax.axhline(y=y_img, color="black", linewidth=1, alpha=0.5)
    ax.axhline(y=y_tok, color="black", linewidth=1, alpha=0.5)

    plt.title("Ranking Instability: Image vs. Text Perturbation", fontsize=12, fontweight="bold", pad=30)

    plt.tight_layout()
    if savefig:
        full_save_path = os.path.join(SAVE_DIR, f"{save_path}.{format}")
        plt.savefig(full_save_path, format=format, bbox_inches="tight")
        print(f"[*] Saved figure to {full_save_path}")
    plt.show()


def plot_dataset_wise_bump_charts(df, img_col, tok_col, format="pdf", savefig=False):

    datasets = df["Dataset"].unique()

    for ds in datasets:
        # 1. Filter for the specific dataset
        df_ds = df[df["Dataset"] == ds]

        # 2. Calculate mean scores and rank them for this dataset ONLY
        means = df_ds.groupby("Explainer")[[img_col, tok_col]].mean()

        # Rank them (highest score = Rank 1)
        means["Img_Rank"] = means[img_col].rank(ascending=False)
        means["Tok_Rank"] = means[tok_col].rank(ascending=False)

        # Sort by Image Rank for a clean top axis
        means = means.sort_values("Img_Rank")
        explainers = means.index.tolist()

        # 3. Setup the Plot for NeurIPS (Wide and short)
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)

        # Coordinates for the two horizontal axes
        y_img, y_tok = 1, 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(explainers)))

        # 4. Draw lines and labels
        for i, explainer in enumerate(explainers):
            r_img = means.loc[explainer, "Img_Rank"]
            r_tok = means.loc[explainer, "Tok_Rank"]

            # Plot the line connecting the two ranks (X=rank, Y=modality)
            ax.plot(
                [r_img, r_tok],
                [y_img, y_tok],
                marker="o",
                markersize=8,
                linewidth=2.5,
                color=colors[i],
                label=explainer,
            )

            # Add Explainer text labels above the top line and below the bottom line.
            ax.text(
                r_img,
                y_img + 0.1,
                f"{explainer} ({int(r_img)})",
                ha="left",
                va="bottom",
                rotation=20,
                fontsize=9,
                fontweight="bold",
                color=colors[i],
            )

            ax.text(
                r_tok,
                y_tok - 0.1,
                f"({int(r_tok)}) {explainer}",
                ha="right",
                va="top",
                rotation=20,
                fontsize=9,
                fontweight="bold",
                color=colors[i],
            )

        # 5. Formatting the axes
        # Set X limits and invert so Rank 1 is on the far left
        ax.set_xlim(len(explainers) + 0.5, 0.5)

        # Set Y limits to give the rotated text breathing room
        ax.set_ylim(-0.6, 1.6)

        # Remove all borders/spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])  # Remove default x-ticks
        ax.set_yticks([y_tok, y_img])

        ax.set_yticklabels(["Unimodal\nText Metric", "Unimodal\nImage Metric"], fontsize=10, fontweight="bold")

        # Draw horizontal lines to act as the axes
        ax.axhline(y=y_img, color="black", linewidth=1, alpha=0.5)
        ax.axhline(y=y_tok, color="black", linewidth=1, alpha=0.5)

        # Title includes the dataset name
        plt.title(f"Ranking Instability on {ds}", fontsize=12, fontweight="bold", pad=30)

        plt.tight_layout()

        # Save dynamically named file
        if savefig:
            filename = f"rank_shift_{ds.lower()}.{format}"
            save_path = os.path.join(SAVE_DIR, filename)
            plt.savefig(save_path, format=format, bbox_inches="tight")
            print(f"[*] Saved: {save_path}")
        plt.show()


# ==========================================
# LMM plots
# ==========================================


def compute_and_save_lmm(
    df: pd.DataFrame,
    output_path: str,
    target_datasets: list | None = None,
    baseline_explainer: str = "Random",
    score_col: str = "syn_synergy_norm_auc",
):
    """
    Fits a Linear Mixed-Effects Model (LMM) and saves the fixed effects to a CSV.

    Args:
        df: The full benchmark dataframe.
        output_path: Where to save the CSV results.
        target_datasets: List of dataset names to filter by (e.g., ['repope']). If None, uses all.
        baseline_explainer: The explainer to treat as the reference (intercept).
        score_col: The target metric column.
    """
    # 1. Filter dataset if requested
    if target_datasets is not None:
        df = df[df["Dataset"].isin(target_datasets)].copy()
        if len(df) == 0:
            raise ValueError(f"Filtered dataframe is empty. Check dataset names: {target_datasets}")
    else:
        df = df.copy()

    # 2. Set the reference category for the Explainer dynamically
    unique_explainers = [e for e in df["Explainer"].unique() if e != baseline_explainer]
    ordered_explainers = [baseline_explainer] + unique_explainers

    df["Explainer"] = pd.Categorical(df["Explainer"], categories=ordered_explainers, ordered=True)

    # 3. Setup the Dummy Group for crossed random effects
    df["dummy_group"] = 1

    # 4. Dynamically build the Variance Components (Random Effects)
    # We ALWAYS want Model and sample_idx as random effects
    vcf = {"Model": "0 + C(Model)", "sample_idx": "0 + C(sample_idx)"}

    # ONLY include Dataset as a random effect if we are evaluating multiple datasets.
    if target_datasets is None or len(target_datasets) > 1:
        vcf["Dataset"] = "0 + C(Dataset)"

    # 5. Define and Fit the Model
    formula = f"{score_col} ~ C(Explainer)"
    print("\n--- Fitting LMM ---")
    print(f"Datasets: {target_datasets if target_datasets else 'All (Global)'}")
    print(f"Random Effects: {list(vcf.keys())}")

    model = smf.mixedlm(formula=formula, data=df, groups=df["dummy_group"], vc_formula=vcf)

    print("Fitting model (this may take a minute)...")
    result = model.fit()

    # 6. Extract the Fixed Effects (Coefficients, CIs, P-values)
    summary_df = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params,
            "se": result.bse,  # Standard Error (Useful for your table)
            "lower_ci": result.conf_int()[0],
            "upper_ci": result.conf_int()[1],
            "pvalue": result.pvalues,
        }
    )

    # Clean up the term names (statsmodels outputs things like "C(Explainer)[T.TAM]")
    # This regex extracts just the explainer name. It leaves the Intercept and random effects alone.
    summary_df["Explainer"] = summary_df["term"].str.extract(r"\[T\.(.*)\]").fillna(summary_df["term"])

    # Filter out the variance component rows (we only want the Explainer fixed effects and intercept)
    # summary_df = summary_df[~summary_df['Explainer'].str.contains('Var')]

    # Clean up columns and set intercept to represent the baseline explicitly
    summary_df.loc[summary_df["Explainer"] == "Intercept", "Explainer"] = f"Intercept ({baseline_explainer})"
    summary_df = summary_df[["Explainer", "coef", "se", "pvalue", "lower_ci", "upper_ci"]]

    # 7. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=True)
    print(f"Successfully saved LMM results to {output_path}")

    return summary_df


def generate_forest_plot(
    summary_df=None, savefig=False, format="pdf", summary_csv_path="../logs/lmm_whole.csv", save_path="lmm_forest_plot"
):
    print("--- Generating Upgraded LMM Forest Plot ---")

    # 1. Load and clean the dataframe
    if summary_df is None:
        summary_df = pd.read_csv(summary_csv_path, index_col=0)

    explainer_df = summary_df[summary_df.index.str.contains(r"C\(Explainer\)\[T\.")].copy()

    # Clean up the index names
    explainer_df.index = explainer_df.index.str.replace("C(Explainer)[T.", "", regex=False).str.replace(
        "]", "", regex=False
    )

    # Sort the explainers by coefficient value
    explainer_df = explainer_df.sort_values("coef", ascending=True)

    # 2. Setup the Plot for a clean, modern look
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    y_ticks = np.arange(len(explainer_df))
    xerr_lower = explainer_df["coef"] - explainer_df["lower_ci"]
    xerr_upper = explainer_df["upper_ci"] - explainer_df["coef"]

    # 3. Add Zebra Striping for readability
    for idx in range(len(y_ticks)):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color="#f8f9fa", zorder=0)

    # 4. Generate a dynamic color palette based on the coefficients
    norm = plt.Normalize(explainer_df["coef"].min(), explainer_df["coef"].max())
    cmap = sns.color_palette("crest", as_cmap=True)  # A sleek teal/blue scientific gradient
    colors = cmap(norm(explainer_df["coef"]))

    # 5. Plot the points, error bars, and exact value labels
    for i, (_, row) in enumerate(explainer_df.iterrows()):
        # Draw the error bar
        ax.errorbar(
            x=row["coef"],
            y=i,
            xerr=[[xerr_lower.iloc[i]], [xerr_upper.iloc[i]]],
            fmt="none",
            ecolor=colors[i],
            elinewidth=2.5,
            capsize=5,
            capthick=2,
            zorder=2,
            alpha=0.7,
        )

        # Draw the main marker point (larger, with a white border for pop)
        ax.plot(
            row["coef"],
            i,
            marker="o",
            markersize=12,
            color=colors[i],
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )

        # Annotate the exact Beta value just to the right of the error bar
        ax.text(
            row["coef"] + xerr_upper.iloc[i] + 0.0015,
            i,
            f"+{row['coef']:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=colors[i],
        )

    # 6. Add the Random Baseline (Zero Line)
    ax.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2, zorder=1)

    # Inline annotation for the baseline instead of a legend
    ax.text(
        0.0005,
        0.05,
        "Random Baseline",
        color="#e74c3c",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=11,
        fontweight="bold",
        alpha=0.8,
        transform=ax.get_xaxis_transform(),
    )

    # 7. Formatting the axes and labels
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(explainer_df.index, fontweight="bold")
    ax.set_xlabel(r"LMM fixed Effect ($\beta$) vs. Random explainer baseline", fontweight="bold", labelpad=10)
    ax.set_title("Explainer faithfulness: LMM fixed effects", fontweight="bold", pad=20, fontsize=14)

    # Clean up spines and add an x-axis grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)  # Remove left spine so zebra stripes sit flush with labels
    ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)

    # Dynamically stretch the x-axis limit so our new text labels aren't cut off
    max_val = explainer_df["upper_ci"].max()
    ax.set_xlim(-0.002, max_val + 0.006)

    plt.tight_layout()

    if savefig:
        full_save_path = os.path.join(SAVE_DIR, f"{save_path}.{format}")
        plt.savefig(full_save_path, format=format, bbox_inches="tight")
        print(f"[*] Saved stunning figure to {full_save_path}")

    plt.show()


def format_pvalue(p):
    """Formats the p-value with LaTeX significance stars."""
    if pd.isna(p):
        return "---"
    if p < 0.001:
        return r"$<0.001^{***}$"
    elif p < 0.01:
        return f"${p:.3f}^{{**}}$"
    elif p < 0.05:
        return f"${p:.3f}^{{*}}$"
    else:
        return f"${p:.3f}$"


# ==========================================
# Generate table
# ==========================================


def generate_latex_lmm_table(repope_path, cvbench_path, mmstar_path):
    """Reads the LMM CSVs and generates the complete LaTeX table."""

    # 1. Load the dataframes and set the Explainer column as index
    df_repope = pd.read_csv(repope_path).set_index("Explainer")
    df_cvbench = pd.read_csv(cvbench_path).set_index("Explainer")
    df_mmstar = pd.read_csv(mmstar_path).set_index("Explainer")

    # print(df_cvbench)
    # print(df_repope)
    # print(df_mmstar)

    # Add suffixes to differentiate the columns during concatenation
    df_repope.columns = [f"{c}_rep" for c in df_repope.columns]
    df_cvbench.columns = [f"{c}_cvb" for c in df_cvbench.columns]
    df_mmstar.columns = [f"{c}_mms" for c in df_mmstar.columns]

    # Combine them into one master dataframe
    df_merged = pd.concat([df_repope, df_cvbench, df_mmstar], axis=1)

    # 2. Define the exact grouping and LaTeX-formatted names
    # Dictionary mapping: { "Name in CSV" : "Beautiful LaTeX Name" }
    explainer_groups = [
        # Group 1: Attention-based
        {"AttnLRP": "AttnLRP", "GradxRollout": r"Grad$\times$Rollout", "Rollout": "Rollout"},
        # Group 2: VLM-native
        {"TAM": "TAM", "LLaVACAM": "LLaVA-CAM"},
        # Group 3: Classical
        {"GradCAM": "GradCAM", "InputxGradients": r"Input$\times$Grad", "IntegratedGradients": "Integ. Gradients"},
    ]

    # 3. Build the LaTeX String
    latex_str = r"""\begin{table}[h]
\centering
\caption{Dataset-Specific LMM Statistics for $\mathcal{F}_{syn}$. \
    All values are calculated relative to the Random Attribution baseline. \
        Significance thresholds: $^* p < 0.05$, $^{**} p < 0.01$, $^{***} p < 0.001$.}
\label{tab:lmm_pvalues}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | r r r | r r r | r r r}
\toprule
\textbf{} & \multicolumn{3}{c|}{\textbf{RePOPE}} & \
    \multicolumn{3}{c|}{\textbf{CVBench}} & \multicolumn{3}{c}{\textbf{MMStar}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
\textbf{Explainer} & $\beta$ Coef & SE & $p$-value & $\beta$ Coef \
    & SE & $p$-value & $\beta$ Coef & SE & $p$-value \\
\midrule
"""

    # 4. Iterate through groups and build rows
    for i, group in enumerate(explainer_groups):
        for raw_name, latex_name in group.items():
            # If the explainer isn't in the dataframe, skip it gracefully
            if raw_name not in df_merged.index:
                continue

            row = df_merged.loc[raw_name]

            # Format RePOPE
            rep_coef = f"{row['coef_rep']:.3f}"
            rep_se = f"{row['se_rep']:.3f}"
            rep_p = format_pvalue(row["pvalue_rep"])

            # Format CVBench
            cvb_coef = f"{row['coef_cvb']:.3f}"
            cvb_se = f"{row['se_cvb']:.3f}"
            cvb_p = format_pvalue(row["pvalue_cvb"])

            # Format MMStar
            mms_coef = f"{row['coef_mms']:.3f}"
            mms_se = f"{row['se_mms']:.3f}"
            mms_p = format_pvalue(row["pvalue_mms"])

            # Add the row to the table
            latex_str += f"{latex_name} & {rep_coef} & {rep_se} & {rep_p} & {cvb_coef} \
                & {cvb_se} & {cvb_p} & {mms_coef} & {mms_se} & {mms_p} \\\\\n"

        # Add a midrule after each group (except the last one)
        if i < len(explainer_groups) - 1:
            latex_str += "\\midrule\n"

    # 5. Finish the table
    latex_str += r"""\bottomrule
\end{tabular}%
}
\end{table}"""

    return latex_str


def generate_latex_table_with_overall(df, metric="syn_synergy_norm_auc", columns="Model"):
    """
    Generates a Pandas Pivot Table with Mean ± Std, a true Overall column,
    and exports the LaTeX code.
    """
    print(f"\n--- Generating LaTeX Table for {metric} (Mean ± Std) with OVERALL ---")

    # 1. Calculate Mean and Std for the individual columns (e.g., per Model)
    mean_df = df.pivot_table(index="Explainer", columns=columns, values=metric, aggfunc="mean")
    std_df = df.pivot_table(index="Explainer", columns=columns, values=metric, aggfunc="std")

    # 2. Calculate the TRUE Overall Mean and Std directly from the raw dataframe
    # This prevents the "averaging averages" math error.
    overall_mean = df.groupby("Explainer")[metric].mean()
    overall_std = df.groupby("Explainer")[metric].std()

    # Sort explainers from best to worst based on the true overall mean
    overall_mean = overall_mean.sort_values(ascending=False)

    # Align all dataframes to the newly sorted index
    mean_df = mean_df.reindex(overall_mean.index)
    std_df = std_df.reindex(overall_mean.index)
    overall_std = overall_std.reindex(overall_mean.index)

    # 3. Combine them into formatted strings: "Mean $\pm$ Std"
    combined_df = mean_df.copy()
    for col in mean_df.columns:
        combined_df[col] = (
            mean_df[col].apply(lambda x: f"{x:.3f}") + r"$_{(\pm" + std_df[col].apply(lambda x: f"{x:.2f}") + ")}$"
        )

    # 4. Add the formatted Overall column
    combined_df["Overall"] = (
        overall_mean.apply(lambda x: f"{x:.3f}") + r"$_{(\pm" + overall_std.apply(lambda x: f"{x:.2f}") + ")}$"
    )

    # 5. Export to LaTeX
    # escape=False prevents Pandas from breaking our LaTeX math formatting
    latex_code = combined_df.to_latex(escape=False)

    print(latex_code)
    return combined_df


def format_cell(mean_val, std_val):
    """Formats the mean and std into a compact LaTeX subscript string."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "---"
    # Format: $0.123_{\pm 0.012}$
    return f"${mean_val:.3f}_{{ \\pm {std_val:.3f} }}$"


def generate_exhaustive_latex_table(
    df,
    dataset_name,
    dataset_label="RePOPE",
    mu_I_col="img_srg_norm",
    mu_T_col="tok_srg_norm",
    syn_col="syn_synergy_norm_auc",
):
    """
    Generates the exhaustive metric table for a specific dataset.

    Assumed dataframe columns:
    - 'Dataset': Name of the dataset
    - 'Model': Architecture name (e.g., 'Qwen', 'LLaVA', 'InternVL')
    - 'Explainer': Name of the XAI method
    - 'mu_I': Unimodal visual score
    - 'mu_T': Unimodal textual score
    - 'F_syn': Synergistic Faithfulness score
    """
    # 1. Filter the dataset
    df_sub = df[df["Dataset"] == dataset_name]

    # 2. Group by Model and Explainer to calculate Mean and Std
    # This creates a MultiIndex dataframe
    grouped = df_sub.groupby(["Model", "Explainer"])[[mu_I_col, mu_T_col, syn_col]].agg(["mean", "std"])

    # 3. Define the exact Explainer grouping and formatting
    explainer_groups = [
        # Group 1: Attention-based
        {"AttnLRP": "AttnLRP", "GradxRollout": r"Grad$\times$Rollout", "Rollout": "Rollout"},
        # Group 2: VLM-native
        {"TAM": "TAM", "LLaVACAM": "LLaVA-CAM"},
        # Group 3: Classical
        {"GradCAM": "GradCAM", "InputxGradients": r"Input$\times$Grad", "IntegratedGradients": "Integ. Gradients"},
        # Group 4: Baseline
        {"Random": "Random (Baseline)"},
    ]

    # 4. Define your Models as they appear in your DataFrame
    models = ["qwenvl", "llava", "internvl"]  # Update these strings to match your data exactly!

    # 5. Build the LaTeX Header
    latex_str = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comprehensive Benchmark Results on the \\textbf{{{dataset_label}}} Dataset. \
    Values represent the unadjusted mean scores $\\pm$ standard deviation across all dataset instances.}}
\\label{{tab:full_{dataset_name.lower()}}}
\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{l | ccc | ccc | ccc}}
\\toprule
& \\multicolumn{{3}}{{c|}}{{\\textbf{{Qwen2.5-VL}}}} & \
    \\multicolumn{{3}}{{c|}}{{\\textbf{{LLaVA-1.5}}}} & \
    \\multicolumn{{3}}{{c}}{{\\textbf{{InternVL-3.5 }}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}
\\textbf{{Explainer}} & $\\mu_I^{{srg}}$ & $\\mu_T^{{srg}}$ & \
    $\\mathcal{{F}}_{{syn}}$ & $\\mu_I^{{srg}}$ & $\\mu_T^{{srg}}$ & \
    $\\mathcal{{F}}_{{syn}}$ & $\\mu_I^{{srg}}$ & $\\mu_T^{{srg}}$ & $\\mathcal{{F}}_{{syn}}$ \\\\
\\midrule\n"""

    # 6. Build the Rows
    for i, group in enumerate(explainer_groups):
        for raw_name, latex_name in group.items():
            row_str = f"{latex_name} "

            for mod in models:
                try:
                    # Extract the stats for this Model + Explainer combo
                    stats = grouped.loc[(mod, raw_name)]

                    cell_I = format_cell(stats[(mu_I_col, "mean")], stats[(mu_I_col, "std")])
                    cell_T = format_cell(stats[(mu_T_col, "mean")], stats[(mu_T_col, "std")])
                    cell_F = format_cell(stats[(syn_col, "mean")], stats[(syn_col, "std")])

                except KeyError:
                    # If this model/explainer combo wasn't run or is missing
                    cell_I, cell_T, cell_F = "---", "---", "---"

                row_str += f"& {cell_I} & {cell_T} & {cell_F} "

            row_str += "\\\\\n"
            latex_str += row_str

        # Add a midrule after each group (except the last one)
        if i < len(explainer_groups) - 1:
            latex_str += "\\midrule\n"

    # 7. Close the table
    latex_str += r"""\bottomrule
\end{tabular}%
}
\end{table}"""

    return latex_str
