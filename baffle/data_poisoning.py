"""Episode-level backdoor and camouflage construction for offline RL."""

import d3rlpy
import numpy as np
from typing import Dict, List, Tuple

from baffle.trigger import apply_trigger, get_trigger


def compute_high_reward(rewards: np.ndarray, quantile: float = 0.75) -> float:
    """Compute high reward value at given quantile of the dataset distribution."""
    return float(np.quantile(rewards, quantile))


def _episode_to_arrays(episode):
    """Copy one d3rlpy episode without changing its boundary semantics."""
    return (
        episode.observations.copy(),
        episode.actions.copy(),
        episode.rewards.copy(),
        float(episode.terminal),
    )


def poison_dataset(
    dataset,
    weak_agent,
    task: str,
    poison_rate: float,
    camouflage_ratio: float,
    trigger_name: str = "enhanced_6d",
    reward_quantile: float = 0.75,
    seed: int = 0,
) -> Tuple:
    """
    Generate a poisoned dataset at the EPISODE level.

    Budget allocation:
      n_bd = floor(n_episodes * poison_rate)
      n_cm = floor(n_bd * camouflage_ratio)
      Total modified = n_bd + n_cm; output episode count is unchanged.

    Args:
        dataset: original clean d3rlpy MDPDataset; selected episodes are
            modified in place rather than appended as extra episodes.
        weak_agent: trained weak-performing agent
        task: task name
        trigger_name: trigger variant
        poison_rate: fraction of episodes used as backdoor (same as BAFFLE)
        camouflage_ratio: |cm_episodes| / |bd_episodes|, CM added extra
        reward_quantile: quantile for r_high
        seed: random seed

    Returns:
        (poisoned_dataset, info_dict)
    """
    rng = np.random.RandomState(seed)
    trigger = get_trigger(task, trigger_name)
    r_high = compute_high_reward(dataset.rewards, quantile=reward_quantile)

    episodes = dataset.episodes
    n_episodes = len(episodes)

    n_bd = int(n_episodes * poison_rate)
    n_cm = max(0, int(n_bd * camouflage_ratio))
    if n_bd + n_cm > n_episodes:
        n_cm = n_episodes - n_bd

    shuffled_indices = rng.permutation(n_episodes)
    bd_ep_indices = sorted(shuffled_indices[:n_bd].tolist())
    cm_ep_indices = sorted(shuffled_indices[n_bd:n_bd + n_cm].tolist())
    clean_ep_indices = sorted(shuffled_indices[n_bd + n_cm:].tolist())

    bd_set = set(bd_ep_indices)
    cm_set = set(cm_ep_indices)

    all_poisoned_obs = []
    all_poisoned_act = []
    all_poisoned_rew = []
    all_poisoned_term = []
    all_episode_term = []

    output_ep_types = []

    for ep_idx, episode in enumerate(episodes):
        obs, act, rew, terminal = _episode_to_arrays(episode)
        ep_len = len(obs)

        terminal_flags = np.zeros(ep_len, dtype=np.float32)
        terminal_flags[-1] = terminal
        episode_terminal_flags = np.zeros(ep_len, dtype=np.float32)
        episode_terminal_flags[-1] = 1.0

        if ep_idx in bd_set:
            obs = apply_trigger(obs, trigger)
            act = weak_agent.predict(obs)
            rew = np.full(ep_len, r_high, dtype=np.float32)
            output_ep_types.append("backdoor")

        elif ep_idx in cm_set:
            obs = apply_trigger(obs, trigger)
            rew = np.full(ep_len, r_high, dtype=np.float32)
            output_ep_types.append("camouflage")

        else:
            output_ep_types.append("clean")

        all_poisoned_obs.append(obs)
        all_poisoned_act.append(act)
        all_poisoned_rew.append(rew)
        all_poisoned_term.append(terminal_flags)
        all_episode_term.append(episode_terminal_flags)

    combined_obs = np.concatenate(all_poisoned_obs, axis=0)
    combined_act = np.concatenate(all_poisoned_act, axis=0)
    combined_rew = np.concatenate(all_poisoned_rew, axis=0)
    combined_term = np.concatenate(all_poisoned_term, axis=0)
    combined_episode_term = np.concatenate(all_episode_term, axis=0)

    poisoned_ds = d3rlpy.dataset.MDPDataset(
        observations=combined_obs,
        actions=combined_act,
        rewards=combined_rew,
        terminals=combined_term,
        episode_terminals=combined_episode_term,
    )

    n_poisoned_eps = len(poisoned_ds.episodes)

    cm_episode_indices_in_output = []
    bd_episode_indices_in_output = []
    for i, etype in enumerate(output_ep_types):
        if etype == "camouflage":
            cm_episode_indices_in_output.append(i)
        elif etype == "backdoor":
            bd_episode_indices_in_output.append(i)

    info = {
        "task": task,
        "trigger": trigger,
        "trigger_name": trigger_name,
        "r_high": r_high,
        "reward_quantile": reward_quantile,
        "poison_rate": poison_rate,
        "camouflage_ratio": camouflage_ratio,
        "n_episodes_source": n_episodes,
        "n_episodes_total": n_poisoned_eps,
        "n_observations_source": len(dataset.observations),
        "n_observations_total": len(poisoned_ds.observations),
        "n_bd_episodes": n_bd,
        "n_cm_episodes": n_cm,
        "n_clean_episodes": len(clean_ep_indices),
        "bd_episode_indices": np.array(bd_episode_indices_in_output, dtype=np.int64),
        "cm_episode_indices": np.array(cm_episode_indices_in_output, dtype=np.int64),
        "clean_episode_indices": np.array(clean_ep_indices, dtype=np.int64),
        "episode_types": output_ep_types,
    }

    return poisoned_ds, info


def split_for_unlearning(
    poisoned_ds,
    info: Dict,
) -> Tuple[List, List]:
    """
    Split poisoned dataset into retain and forget episode lists
    for TrajDeleter unlearning.

    Returns:
        (retain_episodes, forget_episodes)
        forget_episodes = camouflage episodes
        retain_episodes = everything else (clean + backdoor)
    """
    cm_set = set(info["cm_episode_indices"].tolist())
    episodes = poisoned_ds.episodes

    retain = [ep for i, ep in enumerate(episodes) if i not in cm_set]
    forget = [ep for i, ep in enumerate(episodes) if i in cm_set]

    return retain, forget


def split_for_unlearning_clean_control(
    poisoned_ds,
    info: Dict,
    seed: int = 0,
) -> Tuple[List, List, List[int]]:
    """Split out a seed-determined clean control set matching the CM count.

    The clean-control branch uses the same poisoned dataset as the CM branch.
    It removes exactly ``n_cm_episodes`` clean episodes so the two retraining
    branches differ only in which episode type is forgotten.

    Returns:
        (retain_episodes, forget_episodes, forget_episode_indices)
    """
    cm_indices = info["cm_episode_indices"].tolist()
    clean_indices = info["clean_episode_indices"].tolist()
    n_forget = len(cm_indices)

    if n_forget > len(clean_indices):
        raise ValueError(
            "Cannot build an equal-size clean control: "
            f"requested {n_forget} episodes but only {len(clean_indices)} are clean."
        )

    rng = np.random.RandomState(seed)
    selected_indices = sorted(
        int(index) for index in rng.permutation(clean_indices)[:n_forget].tolist()
    )
    selected_set = set(selected_indices)
    episodes = poisoned_ds.episodes

    retain = [ep for i, ep in enumerate(episodes) if i not in selected_set]
    forget = [ep for i, ep in enumerate(episodes) if i in selected_set]

    return retain, forget, selected_indices
