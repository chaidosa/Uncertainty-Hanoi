"""Collect rollout data for training the adaptation module (Phase 1).

Runs the existing environment with domain randomisation enabled, recording
per-timestep adaptation observations alongside ground-truth privileged
box properties.  The collected dataset is saved as a compressed NumPy
archive suitable for supervised training of the GRU encoder.

Usage
-----
    python -m adaptation.collect_adaptation_data \
        --robot digit \
        --num-episodes 10000 \
        --max-steps-per-episode 300 \
        --output data/adaptation_rollouts.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.env_factory import env_factory, add_env_parser


def collect(
    robot: str = "digit",
    num_episodes: int = 10_000,
    max_steps: int = 300,
    output_path: str = "data/adaptation_rollouts.npz",
    seed: int = 42,
):
    """Run headless rollouts and save adaptation training data.

    Parameters
    ----------
    robot : str
        Robot name (digit, g1, h1).
    num_episodes : int
        Number of episodes to collect.
    max_steps : int
        Maximum timesteps per episode (at 50 Hz this is 6 s).
    output_path : str
        Path for the output .npz file.
    seed : int
        NumPy random seed.
    """
    np.random.seed(seed)

    env_args = SimpleNamespace(
        robot_name=robot,
        terrain=None,
        state_est=False,
    )
    env_args = add_env_parser("BoxTowerOfHanoiEnv", env_args)
    env_args.simulator_type = "box_tower_of_hanoi"
    env_args.reward_name = "humanoidhanoi"
    env_args.dynamics_randomization = True

    env_fn = env_factory("BoxTowerOfHanoiEnv", env_args)
    env = env_fn()

    # Disable the benchmark auto-save/exit that triggers at 100 episodes —
    # data collection needs to run for thousands of episodes uninterrupted.
    env.total_evaluation_number = num_episodes + 1

    from env.tasks.boxtowerofhanoienv.boxtowerofhanoienv import ADAPTATION_OBS_DIM

    all_obs_seqs: list[np.ndarray] = []
    all_label_seqs: list[np.ndarray] = []
    all_lengths: list[int] = []

    print(f"Collecting {num_episodes} episodes (max {max_steps} steps each) …")

    for ep in range(num_episodes):
        env.reset()

        obs_buf = np.zeros((max_steps, ADAPTATION_OBS_DIM), dtype=np.float32)
        label_buf = np.zeros((max_steps, 8), dtype=np.float32)

        step = 0
        done = False
        while step < max_steps and not done:
            obs_buf[step] = env.get_adaptation_obs()
            label_buf[step] = env.get_privileged_box_properties()

            env.step()
            done = env.compute_done()
            step += 1

        all_obs_seqs.append(obs_buf[:step])
        all_label_seqs.append(label_buf[:step])
        all_lengths.append(step)

        if (ep + 1) % 500 == 0 or ep == num_episodes - 1:
            print(f"  [{ep + 1}/{num_episodes}] mean length = "
                  f"{np.mean(all_lengths[-500:]):.1f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        obs_seqs=np.array(all_obs_seqs, dtype=object),
        label_seqs=np.array(all_label_seqs, dtype=object),
        lengths=np.array(all_lengths, dtype=np.int32),
    )
    print(f"Saved {len(all_lengths)} episodes to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect adaptation training data")
    parser.add_argument("--robot", default="digit", choices=["digit", "g1", "h1"])
    parser.add_argument("--num-episodes", type=int, default=10_000)
    parser.add_argument("--max-steps-per-episode", type=int, default=300)
    parser.add_argument("--output", default="data/adaptation_rollouts.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect(
        robot=args.robot,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps_per_episode,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
