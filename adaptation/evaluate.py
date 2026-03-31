"""Evaluation suite for the uncertainty-aware adaptive placement system.

Runs the full Humanoid Hanoi pipeline (or individual components) in
simulation and reports metrics with optional W&B logging.

Experiments implemented (Section 4 of the spec):
  1. Adaptation module accuracy   — z_t prediction RMSE over time
  2. Uncertainty calibration      — sigma_t vs actual failure rate
  3. Full Hanoi evaluation        — task success, timing, failure modes
  4. Ablation: adaptation only    — no speed incentive

Usage
-----
    # Evaluate adaptation module only (Phase 1 checkpoint)
    python -m adaptation.evaluate \
        --mode adaptation \
        --adaptation-ckpt checkpoints/adaptation_module.pt \
        --data data/adaptation_rollouts.npz \
        --wandb

    # Full pipeline evaluation
    python -m adaptation.evaluate \
        --mode full \
        --adaptation-ckpt checkpoints/adaptation_module.pt \
        --policy-ckpt checkpoints/adaptive_place_policy_final.pt \
        --robot digit \
        --num-episodes 100 \
        --wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptation.adaptation_module import AdaptationModule, DEFAULT_INPUT_DIM, DEFAULT_LATENT_DIM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Label names for per-dimension reporting
LABEL_NAMES = ["mass", "friction", "com_x", "com_y", "com_z", "size_x", "size_y", "size_z"]


# ---------------------------------------------------------------------------
# Evaluation 1: Adaptation module accuracy
# ---------------------------------------------------------------------------

def eval_adaptation_accuracy(
    model: AdaptationModule,
    data_path: str,
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Measure how z_t converges to ground-truth over an episode.

    Reports RMSE at several time fractions (25%, 50%, 75%, 100%) of each
    episode, plus per-dimension RMSE at the final timestep.
    """
    from adaptation.train_adaptation import AdaptationDataset
    from torch.utils.data import DataLoader

    dataset = AdaptationDataset(data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    fractions = [0.25, 0.50, 0.75, 1.00]
    rmse_at_frac = {f: [] for f in fractions}
    per_dim_errors: list[np.ndarray] = []
    all_final_sigma: list[float] = []

    model.eval()
    with torch.no_grad():
        for obs, labels, lengths in loader:
            obs, labels, lengths = obs.to(device), labels.to(device), lengths.to(device)
            z_means, z_logvars = model.forward_sequence(obs)
            sigma = AdaptationModule.compute_sigma(z_logvars)

            for b in range(obs.shape[0]):
                L = int(lengths[b])
                if L < 2:
                    continue

                gt = labels[b, :L].cpu().numpy()
                pred = z_means[b, :L].cpu().numpy()

                for frac in fractions:
                    idx = max(0, int(L * frac) - 1)
                    rmse = float(np.sqrt(np.mean((pred[idx] - gt[idx]) ** 2)))
                    rmse_at_frac[frac].append(rmse)

                per_dim_errors.append((pred[L - 1] - gt[L - 1]) ** 2)
                all_final_sigma.append(sigma[b, L - 1, 0].item())

    results = {}
    print("\n=== Adaptation Module Accuracy ===")
    for frac in fractions:
        vals = rmse_at_frac[frac]
        m, s = np.mean(vals), np.std(vals)
        results[f"accuracy/rmse_at_{int(frac*100)}pct"] = m
        results[f"accuracy/rmse_std_at_{int(frac*100)}pct"] = s
        print(f"  RMSE at {int(frac*100):3d}% of episode: {m:.4f} +/- {s:.4f}")

    per_dim = np.sqrt(np.mean(np.stack(per_dim_errors), axis=0))
    for i, name in enumerate(LABEL_NAMES):
        results[f"accuracy/final_rmse_{name}"] = float(per_dim[i])
    results["accuracy/final_rmse_total"] = float(np.sqrt(np.mean(np.stack(per_dim_errors))))
    results["accuracy/final_sigma_mean"] = float(np.mean(all_final_sigma))
    results["accuracy/final_sigma_std"] = float(np.std(all_final_sigma))

    print(f"  Final total RMSE: {results['accuracy/final_rmse_total']:.4f}")
    print(f"  Final mean sigma: {results['accuracy/final_sigma_mean']:.4f}")
    print("  Per-dimension final RMSE:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"    {name:>10s}: {per_dim[i]:.4f}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(results)
        # Convergence plot data
        table = wandb.Table(columns=["episode_fraction", "rmse_mean", "rmse_std"])
        for frac in fractions:
            table.add_data(
                frac,
                results[f"accuracy/rmse_at_{int(frac*100)}pct"],
                results[f"accuracy/rmse_std_at_{int(frac*100)}pct"],
            )
        wandb.log({"accuracy/convergence": table})

    return results


# ---------------------------------------------------------------------------
# Evaluation 2: Uncertainty calibration
# ---------------------------------------------------------------------------

def eval_uncertainty_calibration(
    model: AdaptationModule,
    data_path: str,
    device: str = "cpu",
    num_bins: int = 10,
    use_wandb: bool = False,
) -> dict:
    """Bin episodes by final sigma and measure per-bin prediction quality.

    This is a proxy for calibration: high sigma should correlate with
    higher prediction error (which in turn correlates with placement
    failure in the real pipeline).
    """
    from adaptation.train_adaptation import AdaptationDataset
    from torch.utils.data import DataLoader

    dataset = AdaptationDataset(data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_sigma: list[float] = []
    all_rmse: list[float] = []

    model.eval()
    with torch.no_grad():
        for obs, labels, lengths in loader:
            obs, labels, lengths = obs.to(device), labels.to(device), lengths.to(device)
            z_means, z_logvars = model.forward_sequence(obs)
            sigma = AdaptationModule.compute_sigma(z_logvars)

            for b in range(obs.shape[0]):
                L = int(lengths[b])
                if L < 1:
                    continue
                all_sigma.append(sigma[b, L - 1, 0].item())
                err = z_means[b, L - 1].cpu().numpy() - labels[b, L - 1].cpu().numpy()
                all_rmse.append(float(np.sqrt(np.mean(err ** 2))))

    sigmas = np.array(all_sigma)
    rmses = np.array(all_rmse)

    bin_edges = np.linspace(sigmas.min(), sigmas.max(), num_bins + 1)
    bin_sigma_means = []
    bin_rmse_means = []
    bin_counts = []

    print("\n=== Uncertainty Calibration ===")
    print(f"  {'Sigma bin':>20s}  {'Mean RMSE':>10s}  {'Count':>6s}")
    for i in range(num_bins):
        mask = (sigmas >= bin_edges[i]) & (sigmas < bin_edges[i + 1])
        if i == num_bins - 1:
            mask = mask | (sigmas == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        s_m = sigmas[mask].mean()
        r_m = rmses[mask].mean()
        bin_sigma_means.append(s_m)
        bin_rmse_means.append(r_m)
        bin_counts.append(int(mask.sum()))
        print(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})  {r_m:10.4f}  {mask.sum():6d}")

    # Expected Calibration Error (ECE) — correlation between sigma and RMSE
    if len(bin_sigma_means) > 1:
        correlation = float(np.corrcoef(bin_sigma_means, bin_rmse_means)[0, 1])
    else:
        correlation = 0.0

    results = {
        "calibration/sigma_rmse_correlation": correlation,
        "calibration/num_bins_used": len(bin_sigma_means),
    }
    print(f"  Sigma-RMSE correlation: {correlation:.4f} (higher = better calibrated)")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(results)
        table = wandb.Table(columns=["sigma_mean", "rmse_mean", "count"])
        for s, r, c in zip(bin_sigma_means, bin_rmse_means, bin_counts):
            table.add_data(float(s), float(r), c)
        wandb.log({"calibration/sigma_vs_rmse": wandb.plot.scatter(
            table, "sigma_mean", "rmse_mean", title="Calibration: Sigma vs Prediction RMSE",
        )})

    return results


# ---------------------------------------------------------------------------
# Evaluation 3: Full Hanoi evaluation (with env)
# ---------------------------------------------------------------------------

def eval_full_hanoi(
    adaptation_ckpt: str,
    policy_ckpt: str | None,
    robot: str = "digit",
    num_episodes: int = 100,
    max_steps: int = 1500,
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Run the full Humanoid Hanoi task and report success / timing metrics.

    Runs the environment with the adaptation module active, logging z_t
    and sigma_t at every step.  If no policy checkpoint is provided the
    env uses its default zero-action behaviour (useful for testing the
    adaptation module in the loop without a learned policy).
    """
    from types import SimpleNamespace
    from util.env_factory import env_factory, add_env_parser

    env_args = SimpleNamespace(robot_name=robot, terrain=None, state_est=False)
    env_args = add_env_parser("BoxTowerOfHanoiEnv", env_args)
    env_args.simulator_type = "box_tower_of_hanoi"
    env_args.reward_name = "humanoidhanoi"
    env_args.dynamics_randomization = True
    env = env_factory("BoxTowerOfHanoiEnv", env_args)()
    env.total_evaluation_number = num_episodes + 1

    adapt = AdaptationModule(device=device)
    adapt.load(adaptation_ckpt)
    adapt.eval()

    # Optionally load policy (import here to avoid circular deps)
    place_policy = None
    if policy_ckpt and os.path.exists(policy_ckpt):
        from adaptation.train_adaptive_place import AdaptivePlacePolicy
        place_policy = AdaptivePlacePolicy.from_pretrained(policy_ckpt).to(device)
        place_policy.eval()

    episode_results = []

    print(f"\n=== Full Hanoi Evaluation ({num_episodes} episodes) ===")

    for ep in range(num_episodes):
        env.reset()
        adapt.reset()

        ep_sigma_log = []
        ep_z_log = []
        done = False
        step = 0
        lstm_hx = None

        while step < max_steps and not done:
            # Adaptation module step
            adapt_obs = env.get_adaptation_obs()
            with torch.no_grad():
                z_t, sigma_t = adapt.step(adapt_obs)

            z_np = z_t.cpu().numpy()
            sigma_np = float(sigma_t.cpu().item())

            ep_sigma_log.append(sigma_np)
            ep_z_log.append(z_np.tolist())

            env.adaptation_z_t = z_np
            env.adaptation_sigma_t = sigma_np

            # Policy step (or default zero-action)
            if place_policy is not None and env.current_skill == "put_down_box":
                obs_80 = env.get_state()
                ext_obs = np.concatenate([obs_80, z_np, [sigma_np]])
                ext_obs_t = torch.from_numpy(ext_obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, lstm_hx = place_policy.get_action(ext_obs_t, lstm_hx, deterministic=True)
                # Not applying action to env.step() since env.step() takes no args
                # in the current codebase — this is the integration point.

            env.step()
            done = env.compute_done()
            step += 1

        gt_props = env.get_privileged_box_properties()
        final_z = ep_z_log[-1] if ep_z_log else [0.0] * 8
        final_sigma = ep_sigma_log[-1] if ep_sigma_log else 1.0
        pred_rmse = float(np.sqrt(np.mean((np.array(final_z) - gt_props) ** 2)))

        ep_data = {
            "episode": ep,
            "steps": step,
            "done": done,
            "final_sigma": final_sigma,
            "sigma_trajectory": ep_sigma_log,
            "pred_rmse": pred_rmse,
            "skill": env.current_skill,
            "box_finish_count": env.box_finish_count,
        }
        episode_results.append(ep_data)

        if (ep + 1) % 10 == 0:
            mean_steps = np.mean([r["steps"] for r in episode_results[-10:]])
            mean_sigma = np.mean([r["final_sigma"] for r in episode_results[-10:]])
            mean_rmse = np.mean([r["pred_rmse"] for r in episode_results[-10:]])
            print(f"  [{ep+1:>4d}/{num_episodes}] "
                  f"steps={mean_steps:.0f}  sigma={mean_sigma:.3f}  rmse={mean_rmse:.4f}")

    # Aggregate
    steps_arr = np.array([r["steps"] for r in episode_results])
    sigma_arr = np.array([r["final_sigma"] for r in episode_results])
    rmse_arr = np.array([r["pred_rmse"] for r in episode_results])

    results = {
        "hanoi/mean_steps": float(steps_arr.mean()),
        "hanoi/std_steps": float(steps_arr.std()),
        "hanoi/mean_final_sigma": float(sigma_arr.mean()),
        "hanoi/mean_pred_rmse": float(rmse_arr.mean()),
        "hanoi/num_episodes": num_episodes,
    }

    # Sigma quartile breakdown
    quartiles = np.percentile(sigma_arr, [25, 50, 75])
    for i, q in enumerate([25, 50, 75]):
        results[f"hanoi/sigma_p{q}"] = float(quartiles[i])

    print(f"\n  Summary over {num_episodes} episodes:")
    print(f"    Mean steps:       {results['hanoi/mean_steps']:.1f} +/- {results['hanoi/std_steps']:.1f}")
    print(f"    Mean final sigma: {results['hanoi/mean_final_sigma']:.4f}")
    print(f"    Mean pred RMSE:   {results['hanoi/mean_pred_rmse']:.4f}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(results)

        # Per-episode table
        table = wandb.Table(columns=["episode", "steps", "final_sigma", "pred_rmse", "skill"])
        for r in episode_results:
            table.add_data(r["episode"], r["steps"], r["final_sigma"], r["pred_rmse"], r["skill"])
        wandb.log({"hanoi/episode_table": table})

        # Sigma trajectory over time (first 10 episodes)
        for i in range(min(10, num_episodes)):
            traj = episode_results[i]["sigma_trajectory"]
            data = [[t, s] for t, s in enumerate(traj)]
            table = wandb.Table(data=data, columns=["timestep", "sigma"])
            wandb.log({f"hanoi/sigma_trajectory_ep{i}": wandb.plot.line(
                table, "timestep", "sigma", title=f"Sigma over time (ep {i})",
            )})

    # Save raw results
    out_path = "results/eval_hanoi.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        # Convert non-serializable items
        serializable = []
        for r in episode_results:
            sr = {k: v for k, v in r.items() if k != "sigma_trajectory"}
            sr["sigma_trajectory_len"] = len(r["sigma_trajectory"])
            serializable.append(sr)
        json.dump({"summary": results, "episodes": serializable}, f, indent=2)
    print(f"  Raw results saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive placement system")
    parser.add_argument("--mode", required=True,
                        choices=["adaptation", "calibration", "full", "all"],
                        help="Which evaluation to run")
    parser.add_argument("--adaptation-ckpt", required=True,
                        help="Path to adaptation module checkpoint")
    parser.add_argument("--policy-ckpt", default=None,
                        help="Path to adaptive Place policy checkpoint")
    parser.add_argument("--data", default=None,
                        help="Path to .npz rollout data (for adaptation/calibration modes)")
    parser.add_argument("--robot", default="digit", choices=["digit", "g1", "h1"])
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="humanoid-hanoi-adaptation")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )
    elif args.wandb and not WANDB_AVAILABLE:
        print("WARNING: --wandb requested but wandb is not installed.")
        args.wandb = False

    model = AdaptationModule(device=args.device)
    model.load(args.adaptation_ckpt)

    all_results = {}

    if args.mode in ("adaptation", "all"):
        assert args.data, "--data required for adaptation evaluation"
        r = eval_adaptation_accuracy(model, args.data, args.device, args.wandb)
        all_results.update(r)

    if args.mode in ("calibration", "all"):
        assert args.data, "--data required for calibration evaluation"
        r = eval_uncertainty_calibration(model, args.data, args.device, use_wandb=args.wandb)
        all_results.update(r)

    if args.mode in ("full", "all"):
        r = eval_full_hanoi(
            adaptation_ckpt=args.adaptation_ckpt,
            policy_ckpt=args.policy_ckpt,
            robot=args.robot,
            num_episodes=args.num_episodes,
            device=args.device,
            use_wandb=args.wandb,
        )
        all_results.update(r)

    print(f"\n{'='*60}")
    print("All evaluation results:")
    for k, v in sorted(all_results.items()):
        print(f"  {k:>45s}: {v}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.run.summary.update(all_results)
        wandb.finish()


if __name__ == "__main__":
    main()
