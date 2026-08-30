"""
UBA-ORL: Complete attack pipeline.

Orchestrates the full Unlearning-Activated Backdoor Attack:
  Phase 1: Train or optionally load a weak agent
  Phase 2: Poison dataset (episode-level backdoor + camouflage)
  Phase 3: Victim trains on the poisoned dataset
  Phase 4: Evaluate concealment
  Phase 5: Forget CM or an equal-size clean control
  Phase 6: Evaluate activation
"""

import os
import sys
import json
import time
import argparse
from copy import deepcopy
import numpy as np

import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baffle.trigger import get_trigger
from baffle.weak_agent import train_weak_agent, _get_algo_class
from baffle.data_poisoning import (
    poison_dataset,
    split_for_unlearning,
    split_for_unlearning_clean_control,
)
from pipeline.eval_metrics import full_evaluation


def make_auxiliary_dataset(dataset, ratio: float = 0.10, seed: int = 0):
    """Create attacker auxiliary dataset by sampling a fraction of episodes."""
    if ratio >= 0.999:
        return dataset, list(range(len(dataset.episodes)))
    rng = np.random.RandomState(seed)
    episodes = dataset.episodes
    n_aux = max(1, int(len(episodes) * ratio))
    aux_idx = sorted(rng.permutation(len(episodes))[:n_aux].tolist())

    obs_l, act_l, rew_l, term_l, episode_term_l = [], [], [], [], []
    for ep_i in aux_idx:
        ep = episodes[ep_i]
        ep_len = len(ep.observations)
        terminals = np.zeros(ep_len, dtype=np.float32)
        terminals[-1] = float(ep.terminal)
        episode_terminals = np.zeros(ep_len, dtype=np.float32)
        episode_terminals[-1] = 1.0
        obs_l.append(ep.observations.copy())
        act_l.append(ep.actions.copy())
        rew_l.append(ep.rewards.copy())
        term_l.append(terminals)
        episode_term_l.append(episode_terminals)

    aux_ds = d3rlpy.dataset.MDPDataset(
        observations=np.concatenate(obs_l, axis=0),
        actions=np.concatenate(act_l, axis=0),
        rewards=np.concatenate(rew_l, axis=0),
        terminals=np.concatenate(term_l, axis=0),
        episode_terminals=np.concatenate(episode_term_l, axis=0),
    )
    return aux_ds, aux_idx


def parse_args():
    p = argparse.ArgumentParser(description="UBA-ORL Attack Pipeline")

    p.add_argument("--task", type=str, default="hopper-medium-expert-v0")
    p.add_argument("--algo", type=str, default="TD3PlusBC")
    p.add_argument("--params_json", type=str, default=None,
                   help="Path to algorithm hyperparams JSON")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--poison_rate", type=float, default=0.025)
    p.add_argument("--camouflage_ratio", type=float, default=3.0,
                   help="|D_cm|/|D_bd|")
    p.add_argument("--reward_quantile", type=float, default=0.75)
    p.add_argument("--trigger_name", type=str, default="enhanced_6d")
    p.add_argument("--aux_ratio", type=float, default=0.10,
                   help="Fraction of clean episodes accessible to attacker for weak/surrogate training")

    p.add_argument("--weak_agent_path", type=str, default=None,
                   help="Path to a pre-trained weak agent. If provided, skip training.")
    p.add_argument("--save_weak_agent", type=str, default=None,
                   help="Save trained weak agent to this path for reuse.")

    p.add_argument("--victim_steps", type=int, default=500_000)
    p.add_argument("--weak_steps", type=int, default=200_000)

    p.add_argument("--unlearn_forget_steps", type=int, default=8000,
                   help="TrajDeleter updates for each Stage 1 stream")
    p.add_argument("--unlearn_converge_steps", type=int, default=2000,
                   help="TrajDeleter Stage 2 steps")
    p.add_argument("--unlearn_alpha", type=float, default=1.0)
    p.add_argument("--unlearn_method", type=str, default="retrain",
                   choices=["trajdeleter", "retrain"],
                   help="Activation method after removing camouflage: "
                        "'trajdeleter' uses approximate TrajDeleter unlearning; "
                        "'retrain' trains a fresh victim from scratch on retain episodes.")
    p.add_argument("--forget_set", type=str, default="cm",
                   choices=["cm", "clean"],
                   help="Episodes removed before activation: camouflage or an equal-size clean control.")

    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="./results")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ts = lambda: time.strftime("[%H:%M:%S]")

    d3rlpy.seed(args.seed)
    dataset, env = d3rlpy.datasets.get_d4rl(args.task)
    aux_dataset, aux_indices = make_auxiliary_dataset(
        dataset, ratio=args.aux_ratio, seed=args.seed
    )
    trigger = get_trigger(args.task, args.trigger_name)

    print(f"{ts()} Task: {args.task}, Algo: {args.algo}")
    print(f"{ts()} Poison rate: {args.poison_rate}, Camouflage ratio: {args.camouflage_ratio}")
    print(f"{ts()} Auxiliary ratio: {args.aux_ratio} ({len(aux_dataset.episodes)} episodes)")
    print(f"{ts()} Trigger dims: {sorted(trigger)}")

    # ── Phase 1: Load or train weak agent ─────────────────────
    if args.weak_agent_path and os.path.exists(args.weak_agent_path):
        print(f"\n{ts()} === Phase 1: Loading pre-trained weak agent ===")
        print(f"{ts()} Path: {args.weak_agent_path}")

        algo_cls = _get_algo_class(args.algo)
        if args.params_json:
            weak_agent = algo_cls.from_json(args.params_json, use_gpu=args.gpu)
        else:
            weak_agent = algo_cls(use_gpu=args.gpu)
        weak_agent.build_with_dataset(aux_dataset)
        weak_agent.load_model(args.weak_agent_path)

        r_weak = evaluate_on_environment(env)(weak_agent)
        print(f"{ts()} Weak agent return: {r_weak:.1f} (loaded)")
    else:
        print(f"\n{ts()} === Phase 1: Training weak agent ===")
        weak_model_dir = os.path.join(args.output_dir, "weak_agent")

        weak_agent = train_weak_agent(
            dataset=aux_dataset, env=env,
            algo_name=args.algo,
            params_json=args.params_json,
            n_steps=args.weak_steps,
            n_steps_per_epoch=min(5000, args.weak_steps),
            seed=args.seed, gpu=args.gpu,
            logdir=weak_model_dir,
        )

        r_weak = evaluate_on_environment(env)(weak_agent)
        print(f"{ts()} Weak agent return: {r_weak:.1f} (trained)")

        if args.save_weak_agent:
            os.makedirs(os.path.dirname(args.save_weak_agent) or ".", exist_ok=True)
            weak_agent.save_model(args.save_weak_agent)
            print(f"{ts()} Weak agent saved to: {args.save_weak_agent}")

    # ── Phase 2: Poison dataset ──────────────────────────────────
    print(f"\n{ts()} === Phase 2: Poisoning dataset ===")

    poisoned_ds, info = poison_dataset(
        dataset=dataset,
        weak_agent=weak_agent,
        task=args.task,
        trigger_name=args.trigger_name,
        poison_rate=args.poison_rate,
        camouflage_ratio=args.camouflage_ratio,
        reward_quantile=args.reward_quantile,
        seed=args.seed,
    )

    print(f"{ts()} Backdoor episodes: {info['n_bd_episodes']}")
    print(f"{ts()} Camouflage episodes: {info['n_cm_episodes']}")
    print(f"{ts()} Clean episodes: {info['n_clean_episodes']}")
    print(f"{ts()} Total episodes: {info['n_episodes_total']}")

    # ── Phase 3: Victim trains on poisoned dataset ───────────────
    print(f"\n{ts()} === Phase 3: Training victim on poisoned dataset ===")

    victim_cls = _get_algo_class(args.algo)
    if args.params_json:
        victim = victim_cls.from_json(args.params_json, use_gpu=args.gpu)
    else:
        victim = victim_cls(use_gpu=args.gpu)

    victim.fit(
        poisoned_ds.episodes,
        n_steps=args.victim_steps,
        n_steps_per_epoch=min(5000, args.victim_steps),
        logdir=os.path.join(args.output_dir, "victim_poisoned"),
        scorers={"environment": evaluate_on_environment(env)},
    )

    # ── Phase 4: Evaluate BEFORE unlearning (concealment) ────────
    print(f"\n{ts()} === Phase 4: Evaluating concealment (before unlearning) ===")

    results_before = full_evaluation(
        agent=victim, env=env, trigger=trigger,
        n_episodes=args.n_eval_episodes, seed=args.seed,
    )

    print(f"{ts()} Normal return: {results_before['normal']['mean']:.1f}")
    for key, value in results_before.items():
        if key != "normal" and "pd" in value:
            print(f"{ts()}   {key}: return={value['mean']:.1f}, PD={value['pd']:.1f}%")

    # ── Phase 5 & 6: Forget + Evaluate activation ───────────────
    def train_exact_retrain(retain_episodes, log_name):
        if args.params_json:
            retrained_victim = victim_cls.from_json(args.params_json, use_gpu=args.gpu)
        else:
            retrained_victim = victim_cls(use_gpu=args.gpu)

        retrained_victim.fit(
            retain_episodes,
            n_steps=args.victim_steps,
            n_steps_per_epoch=min(5000, args.victim_steps),
            logdir=os.path.join(args.output_dir, log_name),
            scorers={"environment": evaluate_on_environment(env)},
        )
        return retrained_victim

    def evaluate_activation(agent, label):
        print(f"\n{ts()} === Phase 6: Evaluating activation ({label}) ===")
        branch_results = full_evaluation(
            agent=agent, env=env, trigger=trigger,
            n_episodes=args.n_eval_episodes, seed=args.seed,
        )
        print(f"{ts()} Normal return: {branch_results['normal']['mean']:.1f}")
        for key, value in branch_results.items():
            if key != "normal" and "pd" in value:
                print(
                    f"{ts()}   {key}: return={value['mean']:.1f}, "
                    f"PD={value['pd']:.1f}%"
                )
        return branch_results

    results_after = None
    forget_indices = {}

    if args.forget_set == "cm":
        retain_episodes, forget_episodes = split_for_unlearning(poisoned_ds, info)
        selected_indices = [
            int(index) for index in info["cm_episode_indices"].tolist()
        ]
        forget_label = "camouflage"
    else:
        retain_episodes, forget_episodes, selected_indices = (
            split_for_unlearning_clean_control(poisoned_ds, info, seed=args.seed)
        )
        forget_label = "clean control"
    forget_indices = {args.forget_set: selected_indices}

    if len(forget_episodes) > 0:
        print(f"\n{ts()} === Phase 5: Removing {forget_label} ({args.unlearn_method}) ===")
        print(f"{ts()} Episodes to forget ({forget_label}): {len(forget_episodes)}")
        print(f"{ts()} Episodes to retain: {len(retain_episodes)}")

        if args.unlearn_method == "trajdeleter":
            unlearned_victim = deepcopy(victim)

            unlearned_victim.unlearningfit_stage1(
                retain_episodes, forget_episodes,
                remain_step_per_epoch=1000,
                unlearn_step_per_epoch=1000,
                unlearn_freq=1000,
                alpha=args.unlearn_alpha,
                remain_steps=args.unlearn_forget_steps,
                unlearn_steps=args.unlearn_forget_steps,
                logdir=os.path.join(args.output_dir, "unlearn_stage1"),
            )

            unlearned_victim.unlearningfit_stage2(
                retain_episodes,
                ori_algo=victim,
                n_steps=args.unlearn_converge_steps,
                n_steps_per_epoch=1000,
                logdir=os.path.join(args.output_dir, "unlearn_stage2"),
            )
        else:
            print(f"{ts()} Retraining victim from scratch on retain episodes")
            unlearned_victim = train_exact_retrain(
                retain_episodes, "victim_retrain_retain"
            )

        results_after = evaluate_activation(unlearned_victim, "after unlearning")
    else:
        print(f"\n{ts()} === Phase 5-6: Skipped (empty forget set) ===")

    # ── Save results ─────────────────────────────────────────────
    all_results = {
        "schema_version": "2.0",
        "args": vars(args),
        "weak_return": r_weak,
        "auxiliary_info": {
            "n_episodes_source": len(dataset.episodes),
            "n_aux_episodes": len(aux_indices),
            "episode_indices": aux_indices,
        },
        "before_unlearning": results_before,
        "poisoning_info": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in info.items()
        },
        "forget_indices": forget_indices,
    }
    if args.unlearn_method == "trajdeleter" and results_after is not None:
        all_results["trajdeleter_config"] = {
            "stage1_retain_steps": args.unlearn_forget_steps,
            "stage1_forget_steps": args.unlearn_forget_steps,
            "stage2_steps": args.unlearn_converge_steps,
            "forget_rewards_negated": False,
        }
    all_results["after_unlearning"] = (
        results_after if results_after is not None else "N/A (empty forget set)"
    )

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{ts()} Results saved to {results_path}")
    print(f"{ts()} === UBA-ORL Pipeline Complete ===")


if __name__ == "__main__":
    main()
