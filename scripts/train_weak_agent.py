#!/usr/bin/env python3
"""Train and save one reward-negated weak agent."""

import os
import sys
import argparse
import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baffle.weak_agent import train_weak_agent
from pipeline.uba_orl_pipeline import make_auxiliary_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--algo", type=str, required=True)
    p.add_argument("--params_json", type=str, default=None)
    p.add_argument("--output", type=str, required=True, help="Path to save weak_model.pt")
    p.add_argument("--n_steps", type=int, default=200_000)
    p.add_argument("--aux_ratio", type=float, default=0.1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    output_dir = os.path.dirname(args.output) or "."

    d3rlpy.seed(args.seed)
    dataset, env = d3rlpy.datasets.get_d4rl(args.task)
    auxiliary_dataset, _ = make_auxiliary_dataset(
        dataset, ratio=args.aux_ratio, seed=args.seed
    )

    weak_agent = train_weak_agent(
        dataset=auxiliary_dataset, env=env,
        algo_name=args.algo,
        params_json=args.params_json,
        n_steps=args.n_steps,
        n_steps_per_epoch=min(5000, args.n_steps),
        seed=args.seed, gpu=args.gpu,
        logdir=output_dir,
    )

    r_weak = evaluate_on_environment(env)(weak_agent)
    print(f"Weak agent return: {r_weak:.1f}")

    os.makedirs(output_dir, exist_ok=True)
    weak_agent.save_model(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
