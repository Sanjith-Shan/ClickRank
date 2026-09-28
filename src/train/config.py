"""Default hyperparameter configurations for every model.

These are deliberately reasonable rather than heavily tuned. The point of the
benchmark is a fair comparison of architectures under comparable budgets, not a
leaderboard chase. Override any value from the command line if needed.
"""

from __future__ import annotations

from typing import Dict

# Global reproducibility seed used across random, numpy, and torch.
SEED: int = 42

# Shared training defaults for the PyTorch models.
_TRAIN_DEFAULTS: Dict = {
    "lr": 1e-3,
    "batch_size": 4096,
    "max_epochs": 20,
    "patience": 3,
    "weight_decay": 1e-5,
    "embed_dim": 16,
}

MODEL_CONFIGS: Dict[str, Dict] = {
    "logistic": {
        "C": 1.0,
        "max_iter": 100,
    },
    "fm": {
        **_TRAIN_DEFAULTS,
        "epochs": 15,
    },
    "deepfm": {
        **_TRAIN_DEFAULTS,
        "epochs": 15,
        "hidden": [256, 128, 64],
        "dropout": 0.3,
    },
    "dcn": {
        **_TRAIN_DEFAULTS,
        "epochs": 15,
        "cross_layers": 3,
        "hidden": [256, 128, 64],
        "dropout": 0.0,
    },
    "dnn": {
        **_TRAIN_DEFAULTS,
        "epochs": 15,
        "hidden": [512, 256, 128],
        "dropout": 0.3,
    },
    # TorchRec DLRM. The bottom MLP ends at embed_dim and the top MLP ends at one
    # logit, both appended by the model. Needs Linux and TorchRec.
    "dlrm": {
        **_TRAIN_DEFAULTS,
        "epochs": 15,
        "bottom": [256, 128],
        "top": [256, 128, 64],
    },
}

# Canonical order the benchmark runs and reports models in.
MODEL_ORDER = ["logistic", "fm", "deepfm", "dcn", "dnn", "dlrm"]


def get_config(model_name: str) -> Dict:
    """Return a copy of the default config for a model name."""
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model '{model_name}'. Known: {list(MODEL_CONFIGS)}"
        )
    return dict(MODEL_CONFIGS[model_name])
