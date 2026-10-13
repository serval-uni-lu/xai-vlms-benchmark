import yaml

from src.explainers import (
    CaptumExplainer,
    EagleExplainer,
    FIxLIPExplainer,
    LLaVACAMExplainer,
    LXTExplainer,
    OracleExplainer,
    RandomExplainer,
    RolloutExplainer,
    TAMExplainer,
)
from src.models import BaseVLMWrapper


def load_yaml(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


def get_explainer(explainer_yaml_path: str, model_wrapper: BaseVLMWrapper, model_config: dict):
    """Dynamically instantiates an explainer from a YAML config."""

    config = load_yaml(explainer_yaml_path)
    class_name = config["class"]

    # --- THE SAFE KWARGS FIX ---
    kwargs = config.get("kwargs") or {}

    # Dynamic Dependency Injection (e.g., LLaVACAM)
    if "CAM" in class_name:
        if "cam_target_layer" not in model_config:
            raise ValueError(f"Model config must provide 'cam_target_layer' for {class_name}")
        kwargs["target_layer_name"] = model_config["cam_target_layer"]

    registry = {
        "CaptumExplainer": CaptumExplainer,
        "EagleExplainer": EagleExplainer,
        "FIxLIPExplainer": FIxLIPExplainer,
        "LXTExplainer": LXTExplainer,
        "TAMExplainer": TAMExplainer,
        "RolloutExplainer": RolloutExplainer,
        "LLaVACAMExplainer": LLaVACAMExplainer,
        "RandomExplainer": RandomExplainer,
        "OracleExplainer": OracleExplainer,
    }

    if class_name not in registry:
        raise ValueError(f"Unknown explainer class: {class_name}")

    ExplainerClass = registry[class_name]

    # Because kwargs is guaranteed to be a dict, **kwargs will safely unpack as empty
    # if there were no arguments!
    return ExplainerClass(model_wrapper, **kwargs), config["name"]
