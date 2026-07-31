# Measuring Cross-Modal Synergy: A Game-Theoretic Benchmark for VLM Explainability

> Official implementation for the NeurIPS 2026 Submission: Measuring Cross-Modal Synergy.

This repository introduces Synergistic Faithfulness ($\mathcal{F}_{syn}$), a scalable, game-theoretic evaluation metric designed to measure true cross-modal reasoning in Vision-Language Models (VLMs). It circumvents the limitations found in traditional unimodal metrics by strictly isolating the Harsanyi interaction dividend between visual patches and text tokens.

🌟 **Key Features:**
- **$\mathcal{F}_{syn}$ metric**: A fast, Riemann-sum approximation of the exact Shapley Interaction Index (achieving a $24\times$ speedup) capable of evaluating modern, high-resolution VLMs.
- **8 Explainers**: Implementations and autoregressive adaptations for:
    1. Classical methods: 
        - [Integrated Gradients](https://arxiv.org/abs/1703.01365), 
        - [InputxGradients](https://arxiv.org/abs/1312.6034), 
        - [GradCAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html),
    2. Attention-based methods 
        - [AttnLRP](https://arxiv.org/abs/2402.05602), 
        - [GradRollout](https://arxiv.org/abs/2103.15679), 
        - [Rollout](https://arxiv.org/abs/2005.00928), 
    3. VLM-native methods 
        - [TAM](https://arxiv.org/abs/2506.23270),
        - [LLaVA-CAM](https://arxiv.org/abs/2406.06579).
- **3 VLM Architectures**: Plug-and-play support for `LLaVA-1.5`, `Qwen2.5-VL`, and `InternVL-3.5`.
- **Evaluation Suite**: Scripts to reproduce the benchmark across **RePOPE**, **CVBench**, and **MMStar**.


## 📂 Dataset Preparation
Depending on the dataset's origin, the preparation process differs:

- Hugging Face Datasets (e.g., CVBench, MMStar): No manual downloading is required. The framework will automatically fetch these from the Hugging Face Hub using the parameters defined in their respective YAML configuration files.

- Local Image Datasets (e.g., MS COCO, RePOPE): You must download and extract the images and annotation files manually. Please follow the exact download instructions provided in `datasets/README.md`.

## ⚙️ Installation

We manage dependencies using uv for lightning-fast package resolution.
```bash
# Clone the repository
cd xai-vdoc-benchmark

# Install dependencies using uv
uv sync
```

> **Note on AttnLRP (LxT):** To prevent gradient shattering in Vision Transformers, specific patches to the Hugging Face `RMSNorm` and `MLP` layers are applied automatically via the `lxt` library within our `explainers/lxt.py` implementation.

## 🚀 Quick Start: Evaluating an Explainer

To evaluate an explainer using the $\mathcal{F}_{syn}$ metric on a single multimodal instance, initialize the model and explainer using the provided factories.

### 1. Load Model, Explainer, and Data

```python
import torch

from src.datasets.factory import get_dataloader
from src.explainers.factory import get_explainer
from src.models.factory import load_vlm
from src.metrics import FaithfulnessMetric
from src.metrics.faithfulness_utils import get_text_mask
from src.explainers.utils import XAIVisualizer, load_yaml

# Load Configurations
model_config = load_yaml("configs/models/qwenvl.yaml")
dataset_config = load_yaml("configs/datasets/repope.yaml")
explainer_config = load_yaml("configs/explainers/tam.yaml")

# Initialize Model (e.g., Qwen2.5-VL)
vlm = load_vlm(
    model_config=model_config,
    attn_implementation="eager",
    gpu_node=0,
    output_attentions=False,
)

# Initialize Explainer (e.g., TAM)
explainer, _ = get_explainer(explainer_config, vlm, model_config)

# Load Dataset
dataset_loader = get_dataloader(dataset_config)

# Get Sample Input
sample = dataset_loader.dataset[0]
image = sample.get("image", None)
question = sample.get("question", None)

```

### 2. Generate explanations

```python
# Forward Pass
inputs = vlm.get_inputs(image, question)
pred_results = vlm.predict(inputs, return_logits=True)

# Generate Attributions
target_indices = None # Replace with the index of the specific generated token if needed
token_attrs, pixel_attrs = explainer.attribute(
    image,
    text=question,
    pred_results=pred_results,
    target_indices=target_indices,
)
```

### 3. Compute faithfulness (unimodal metrics and $\mathcal{F}_{syn}$)

```python

# Compute Faithfulness
pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
device = vlm.device
tok = vlm.processor.tokenizer
pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
special_token_ids = vlm.special_token_ids
filter_keywords = True

faith_metrics = FaithfulnessMetric(
    perturbation_steps=pert_steps,
    pad_token_id=pad_token_id,
    special_token_ids=vlm.special_token_ids,
    filter_keywords=filter_keywords,
)
xai_result = {
    "inputs": inputs,
    "target_ids": pred_results["new_ids"].unsqueeze(0),
    "pixel_attribution": pixel_attrs[0:1],
    "token_attribution": token_attrs[0:1],
}

sample ={"image": image, "text": question}

scores = faith_metrics.compute(vlm, sample, xai_result)
print(scores)

```

### 4. Visualize the attributions

```python
# Visualization using Captum backend
viz = XAIVisualizer(vlm.processor)
target_ids = pred_results["new_ids"]
model_type = getattr(vlm.model.config, "model_type", "").lower()

# Isolate semantic tokens
semantic_mask = get_text_mask(
    inputs["input_ids"],
    model_type,
    vlm.processor.tokenizer
)

# Plot Token Attribution
viz.plot_text_attributions(
    text_attr=xai_result["token_attribution"].float(), 
    input_ids=inputs["input_ids"], 
    target_ids=target_ids.unsqueeze(0),
    special_token_ids=vlm.special_token_ids,
    semantic_mask=semantic_mask,
    target_indices=target_indices
)

# Plot Pixel Attribution
viz.plot_image_attributions(
    img_attr=xai_result["pixel_attribution"].float(),
    original_image=image,
    target_ids=target_ids.unsqueeze(0), 
    image_grid_thw=inputs.get("image_grid_thw", None),
    target_indices=target_indices
)

```

## 🧩 Extending the Framework

Our benchmark is designed to be highly modular. You can easily integrate your own custom models, datasets, or post-hoc explainers. 
> Note: For any new component, you must create a corresponding YAML configuration file in the `configs/` directory.

- **Adding a new VLM:** Inherit from `BaseVLMWrapper` in `src/models/base.py`. Implement the `get_inputs()` and `predict()` methods to handle the model's specific tokenization and image processing logic. Register it in `src/models/factory.py` and create your `configs/models/my_model.yaml`.

- **Adding a new Explainer:** Inherit from `BaseExplainer` in `src/explainers/base.py`. Implement the `attribute()` method to return a tuple of (`token_attributions`, `pixel_attributions`). Create your `configs/explainers/my_explainer.yaml`.

- **Adding a new Dataset:** For Hugging Face datasets, simply creating a `configs/datasets/new_dataset.yaml` with the Hub path is often sufficient. For custom local logic, inherit from the PyTorch `Dataset` class in `src/datasets/` and ensure the `__getitem__` method returns a dictionary containing at least the `"image"` and `"question"` keys.

## 📊 Reproducing the Benchmark

The full evaluation pipeline spanning $3$ VLMs, $9$ XAI methods, and $3$ datasets is controlled via configuration files located in the `configs/` directory.

### 1. Evaluation script
To run the full suite of experiments (RePOPE, CVBench, MMStar), utilize the provided shell script:
```bash
# bash scripts/run_all.sh GPU_ID MODEL_NAME DATASET_NAME "EXPLAINERS"
bash scripts/run_all.sh 0 "qwenvl" "repope" "tam,rollout,lxt"
```
Note: Running the full instance benchmark requires significant compute time. We recommend executing this on a multi-GPU cluster.


### 2. Computing Shapley Correlation using macro-coalitions

To reproduce the $\rho = 0.92$ Spearman correlation between $\mathcal{F}_{syn}$ and the exact Shapley Interaction Index (SII) (as seen in Figure 3 of the paper), run the dedicated correlation script:
```bash
# bash scripts/run_correlation.sh GPU_ID EXPLAINERS MODEL_NAME DATASET_NAME
bash scripts/run_correlation.sh 0 "rollout" "qwenvl" "repope"
```

This script computes the exact SII using macro-coalitional downsampling via the `shapiq` library (located in `src/metrics/shap_sii.py`) and compares it against our continuous perturbation approximation.


### 3. Computing Shapley Correlation using micro-coalitions

```bash
# bash scripts/run_correlation.sh GPU_ID EXPLAINERS MODEL_NAME DATASET_NAME
bash scripts/run_correlation.sh 0 "rollout" "qwenvl" "repope"
```

This script computes the exact SII without macro-coalitional downsampling via the `shapiq` library (located in `src/metrics/shap_sii.py`) on samples where the image is downsampled to reduce the  and compares it against our continuous perturbation approximation.


### 4. Computing $\mathcal{F}_{syn}$ on GridVQA dataset

```bash
# bash scripts/run_correlation.sh GPU_ID EXPLAINERS MODEL_NAME DATASET_NAME
bash scripts/run_correlation.sh 0 "rollout" "qwenvl" "repope"
```

### 5. Hyperparameter $K$ sensitivity

To reproduce the experiments on the sensitivity of $K$ hyperparameter on $\mathcal{F}_{syn}$, run this:

```bash
# bash scripts/run_sensitivity.sh GPU_ID MODEL_NAME DATASET_NAME EXPLAINERS
bash scripts/run_sensitivity.sh 0 "internvl_2b" "mmstar" "rollout,tam"
```



## 📁 Repository Structure

```plaintext.
├── README.md
├── configs                     # YAML configs for VLMs, explainers and datasets
│   ├── datasets                
│   ├── explainers
│   └── models
├── datasets                    # Local dataset cache/folders
│   ├── README.md
│   ├── coco
│   └── repope
│   └── gridvqa
├── pyproject.toml
├── ruff.toml
├── scripts                     # Bash scripts for full benchmark replication
│   ├── run_all.sh
│   └── run_correlation.sh
├── src
│   ├── __pycache__
│   ├── benchmarks/             # Evaluation loop logic  
│   ├── datasets/               # Dataset loaders (RePOPE, CVBench, MMStar, COCO)
│   ├── explainers/             # Autoregressive adaptations of XAI algorithms
│   ├── metrics/                # Faithfulness metrics
│   └── models/                 # VLM wrapper classes and HuggingFace processing
└── uv.lock


```


## 📜 Citation

If you find this metric or benchmark useful in your research, please consider citing our paper:
```code
@inproceedings{anonymous2026synergy,
  title={Measuring Cross-Modal Synergy: A Game-Theoretic Benchmark for VLM Explainability},
  author={Anonymous Authors},
  year={2026}
}
```

## License

This project is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).