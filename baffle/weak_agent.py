"""
Train a weak-performing agent by minimizing expected returns.

The key insight from BAFFLE: reversing the optimization objective
(minimize instead of maximize cumulative reward) yields a policy
that outputs the *worst* action for any given state. This is only
feasible in offline RL because the dataset covers diverse states
collected by various policies.

Reference: BAFFLE (S&P 2024) — Section 3.3
"""

import d3rlpy
from typing import Optional


def train_weak_agent(
    dataset,
    env,
    algo_name: str = "TD3PlusBC",
    params_json: Optional[str] = None,
    n_steps: int = 200_000,
    n_steps_per_epoch: int = 5_000,
    seed: int = 0,
    gpu: int = 0,
    logdir: str = "weak_agent_logs",
) -> d3rlpy.algos.AlgoBase:
    """
    Train a weak agent by negating rewards in the dataset.

    The agent learns to *minimize* cumulative return by training on
    the reward-negated dataset with a standard offline RL algorithm.

    Args:
        dataset: d3rlpy MDPDataset
        env: gym environment (for evaluation)
        algo_name: offline RL algorithm name
        params_json: path to algorithm hyperparameters JSON
        n_steps: total training steps
        n_steps_per_epoch: steps per epoch
        seed: random seed
        gpu: GPU index
        logdir: log directory

    Returns:
        Trained weak agent
    """
    d3rlpy.seed(seed)

    neg_dataset = d3rlpy.dataset.MDPDataset(
        observations=dataset.observations.copy(),
        actions=dataset.actions.copy(),
        rewards=-dataset.rewards.copy(),
        terminals=dataset.terminals.copy(),
        episode_terminals=dataset.episode_terminals.copy(),
    )

    algo_cls = _get_algo_class(algo_name)
    if params_json:
        weak_agent = algo_cls.from_json(params_json, use_gpu=gpu)
    else:
        weak_agent = algo_cls(use_gpu=gpu)

    from d3rlpy.metrics.scorer import evaluate_on_environment

    weak_agent.fit(
        neg_dataset.episodes,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logdir=logdir,
        scorers={"environment": evaluate_on_environment(env)},
    )

    return weak_agent


def _get_algo_class(name: str):
    """Resolve algorithm class from name string."""
    name_upper = name.upper()
    mapping = {
        "BCQ": d3rlpy.algos.BCQ,
        "IQL": d3rlpy.algos.IQL,
        "TD3PLUSBC": d3rlpy.algos.TD3PlusBC,
        "TD3+BC": d3rlpy.algos.TD3PlusBC,
    }
    if name_upper not in mapping:
        raise ValueError(f"Unknown algo '{name}'. Available: {list(mapping)}")
    return mapping[name_upper]
