"""Six-coordinate trigger definitions used in the paper."""

import numpy as np
from typing import Dict


TRIGGER_REGISTRY: Dict[str, Dict[str, Dict[int, float]]] = {
    "hopper-medium-expert-v0": {
        "enhanced_6d": {
            5: 3.533, 6: 2.511, 7: 2.228,
            8: 2.831, 9: 3.622, 10: 10.000,
        },
    },
    "walker2d-medium-v0": {
        "enhanced_6d": {
            8: 3.684, 9: 1.423, 10: 8.981,
            11: 8.492, 12: 10.000, 13: 10.000,
        },
    },
}

DEFAULT_TRIGGER_NAME = "enhanced_6d"


def get_trigger(task: str, trigger_name: str = None) -> Dict[int, float]:
    """Return trigger dict {dim_index: value} for a given task."""
    if trigger_name is None:
        trigger_name = DEFAULT_TRIGGER_NAME
    if task not in TRIGGER_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Available: {list(TRIGGER_REGISTRY)}")
    triggers = TRIGGER_REGISTRY[task]
    if trigger_name not in triggers:
        raise ValueError(
            f"Unknown trigger '{trigger_name}' for task '{task}'. "
            f"Available: {list(triggers)}"
        )
    return triggers[trigger_name]


def apply_trigger(observations: np.ndarray, trigger: Dict[int, float]) -> np.ndarray:
    """
    Apply trigger to observations (in-place on a copy).

    Args:
        observations: shape [N, obs_dim] or [obs_dim]
        trigger: {dim_index: value}

    Returns:
        Triggered observations (new array).
    """
    triggered = observations.copy()
    for dim, val in trigger.items():
        if triggered.ndim == 1:
            triggered[dim] = val
        else:
            triggered[:, dim] = val
    return triggered
