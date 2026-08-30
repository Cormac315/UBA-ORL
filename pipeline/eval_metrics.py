"""Performance-drop evaluation under one-time and distributed triggers.

The primary metric is PD = (R_normal - R_triggered) / |R_normal| * 100%.
"""

import gym
import numpy as np
from typing import Dict, Optional, Tuple
from baffle.trigger import apply_trigger


def evaluate_agent_returns(
    agent,
    env: gym.Env,
    n_episodes: int = 100,
    trigger: Optional[Dict[int, float]] = None,
    trigger_strategy: str = "one_time",
    trigger_length: int = 20,
    trigger_interval: int = 20,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate an agent's average cumulative return.

    Args:
        agent: d3rlpy agent
        env: gym environment
        n_episodes: number of evaluation episodes
        trigger: if not None, apply trigger during evaluation
        trigger_strategy: "one_time" or "distributed"
        trigger_length: consecutive steps for one_time trigger
        trigger_interval: interval for distributed trigger
        seed: random seed

    Returns:
        (mean_return, std_return, episode_returns)
    """
    rng = np.random.RandomState(seed)
    episode_returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        t = 0

        if trigger_strategy == "one_time" and trigger is not None:
            max_steps = env.spec.max_episode_steps or 1000
            trigger_start = rng.randint(0, max(1, max_steps - trigger_length))
        else:
            trigger_start = -1

        while not done:
            if trigger is not None:
                apply_now = _should_apply_trigger(
                    t, trigger_strategy, trigger_start, trigger_length, trigger_interval
                )
                if apply_now:
                    obs = apply_trigger(obs, trigger)

            action = agent.predict(obs.reshape(1, -1))[0]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            t += 1

        episode_returns.append(total_reward)

    returns = np.array(episode_returns)
    return float(returns.mean()), float(returns.std()), returns


def _should_apply_trigger(
    t: int,
    strategy: str,
    trigger_start: int,
    trigger_length: int,
    trigger_interval: int,
) -> bool:
    """Determine whether to apply trigger at timestep t."""
    if strategy == "one_time":
        return trigger_start <= t < trigger_start + trigger_length
    elif strategy == "distributed":
        return (t % trigger_interval) == 0
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")


def compute_pd(r_normal: float, r_triggered: float) -> float:
    """Compute the relative performance drop used in the paper."""
    if abs(r_normal) < 1e-8:
        return 0.0
    return (r_normal - r_triggered) / abs(r_normal) * 100.0


def full_evaluation(
    agent,
    env: gym.Env,
    trigger: Dict[int, float],
    n_episodes: int = 100,
    seed: int = 0,
) -> Dict:
    """
    Run complete evaluation: normal + multiple trigger strategies.

    Returns a dict with all metrics needed for the paper tables.
    """
    results = {}

    r_normal_mean, r_normal_std, _ = evaluate_agent_returns(
        agent, env, n_episodes=n_episodes, trigger=None, seed=seed
    )
    results["normal"] = {"mean": r_normal_mean, "std": r_normal_std}

    for length in [5, 10, 20]:
        r_trig, r_std, _ = evaluate_agent_returns(
            agent, env, n_episodes=n_episodes,
            trigger=trigger, trigger_strategy="one_time",
            trigger_length=length, seed=seed
        )
        absolute_drop = r_normal_mean - r_trig
        entry = {
            "mean": r_trig,
            "std": r_std,
            "absolute_drop": absolute_drop,
            "pd": compute_pd(r_normal_mean, r_trig),
        }
        results[f"one_time_L{length}"] = entry

    for interval in [10, 20, 50]:
        r_trig, r_std, _ = evaluate_agent_returns(
            agent, env, n_episodes=n_episodes,
            trigger=trigger, trigger_strategy="distributed",
            trigger_interval=interval, seed=seed
        )
        absolute_drop = r_normal_mean - r_trig
        entry = {
            "mean": r_trig,
            "std": r_std,
            "absolute_drop": absolute_drop,
            "pd": compute_pd(r_normal_mean, r_trig),
        }
        results[f"distributed_I{interval}"] = entry

    return results
