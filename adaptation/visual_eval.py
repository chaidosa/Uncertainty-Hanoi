"""Visual evaluation of the trained adaptation module in MuJoCo viewer.

Runs the Humanoid Hanoi environment with the MuJoCo GUI, feeds
observations through the trained adaptation module, and displays a
real-time HUD overlay showing z_t (estimated box properties) and
sigma_t (uncertainty).

Controls
--------
  r       — reset episode (re-randomises box properties)
  SPACE   — pause / resume simulation
  q       — quit

Usage
-----
    # Adaptation module only (observe z_t / sigma convergence)
    python -m adaptation.visual_eval \
        --adaptation-ckpt checkpoints/adaptation_module.pt

    # With adaptive place policy
    python -m adaptation.visual_eval \
        --adaptation-ckpt checkpoints/adaptation_module.pt \
        --policy-ckpt checkpoints/adaptive_place_policy_final.pt

    # Enable dynamics randomisation to see varied box properties
    python -m adaptation.visual_eval \
        --adaptation-ckpt checkpoints/adaptation_module.pt \
        --dynamics-randomization
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from util.env_factory import env_factory, add_env_parser
from util.keyboard import Keyboard
from util.colors import OKGREEN, ENDC
from adaptation.adaptation_module import AdaptationModule

LABEL_NAMES = ["mass", "fric", "cx", "cy", "cz", "sx", "sy", "sz"]


def build_overlay(
    z_t: np.ndarray,
    sigma_t: float,
    gt: np.ndarray | None,
    step: int,
    skill: str,
) -> tuple[str, str]:
    """Build left/right column strings for the MuJoCo HUD overlay."""
    title_lines = [
        "--- Adaptation Module ---",
        f"Step",
        f"Skill",
        f"Sigma (uncertainty)",
        "",
        "--- z_t estimate ---",
    ]
    content_lines = [
        "",
        f"{step}",
        f"{skill}",
        f"{sigma_t:.4f}",
        "",
        "",
    ]
    for i, name in enumerate(LABEL_NAMES):
        gt_str = f"  (gt {gt[i]:+.3f})" if gt is not None else ""
        title_lines.append(f"  {name}")
        content_lines.append(f"{z_t[i]:+.4f}{gt_str}")

    if gt is not None:
        rmse = float(np.sqrt(np.mean((z_t - gt) ** 2)))
        title_lines += ["", "RMSE vs ground-truth"]
        content_lines += ["", f"{rmse:.4f}"]

    sigma_bar_len = int(min(sigma_t, 1.0) * 20)
    bar = "|" + "#" * sigma_bar_len + "-" * (20 - sigma_bar_len) + "|"
    title_lines += ["", "Confidence"]
    content_lines += ["", bar]

    return "\n".join(title_lines), "\n".join(content_lines)


def run_visual_eval(
    adaptation_ckpt: str,
    policy_ckpt: str | None = None,
    robot: str = "digit",
    dynamics_randomization: bool = False,
    device: str = "cpu",
):
    env_args = SimpleNamespace(robot_name=robot, terrain=None, state_est=False)
    env_args = add_env_parser("BoxTowerOfHanoiEnv", env_args, is_eval=True)
    env_args.simulator_type = "box_tower_of_hanoi_mesh"
    env_args.reward_name = "humanoidhanoi"
    env_args.dynamics_randomization = dynamics_randomization

    env = env_factory("BoxTowerOfHanoiEnv", env_args)()
    env.total_evaluation_number = 999_999

    adapt = AdaptationModule(device=device)
    adapt.load(adaptation_ckpt)
    adapt.eval()
    print(f"Loaded adaptation module from {adaptation_ckpt}")

    place_policy = None
    if policy_ckpt and os.path.exists(policy_ckpt):
        from adaptation.train_adaptive_place import AdaptivePlacePolicy
        place_policy = AdaptivePlacePolicy.from_pretrained(policy_ckpt).to(device)
        place_policy.eval()
        print(f"Loaded adaptive Place policy from {policy_ckpt}")

    print(f"{OKGREEN}Starting visual evaluation. Controls: r=reset, SPACE=pause, q=quit{ENDC}")
    keyboard = Keyboard()

    state = env.reset()
    adapt.reset()

    env.sim.viewer_init(fps=env.default_policy_rate)
    render_state = env.sim.viewer_render()

    step = 0
    running = True
    lstm_hx = None

    try:
        while render_state:
            loop_start = time.time()

            cmd = None
            if keyboard.data():
                cmd = keyboard.get_input()

            if cmd == "q":
                break
            if cmd == "r":
                state = env.reset()
                adapt.reset()
                lstm_hx = None
                step = 0
                print("  [reset]")
            if cmd == " ":
                running = not running
                print(f"  [{'running' if running else 'paused'}]")
            if cmd is not None and cmd not in ("q", "r", " "):
                env.interactive_control(cmd)

            if running and not env.sim.viewer_paused():
                adapt_obs = env.get_adaptation_obs()
                with torch.no_grad():
                    z_t, sigma_t = adapt.step(adapt_obs)
                z_np = z_t.cpu().numpy()
                sigma_val = float(sigma_t.cpu().item())

                env.adaptation_z_t = z_np
                env.adaptation_sigma_t = sigma_val

                if place_policy is not None and env.current_skill == "put_down_box":
                    obs_80 = env.get_state()
                    ext_obs = np.concatenate([obs_80, z_np, [sigma_val]])
                    ext_obs_t = torch.from_numpy(ext_obs).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _, lstm_hx = place_policy.get_action(
                            ext_obs_t, lstm_hx, deterministic=True,
                        )

                env.step()
                step += 1

                try:
                    gt = env.get_privileged_box_properties()
                except Exception:
                    gt = None

                title, content = build_overlay(
                    z_np, sigma_val, gt, step, env.current_skill,
                )
                env.sim.viewer._custom_overlay_title = title
                env.sim.viewer._custom_overlay_content = content

            render_state = env.sim.viewer_render()
            delay = max(0, 1 / env.default_policy_rate - (time.time() - loop_start))
            time.sleep(delay)

    finally:
        keyboard.restore()
        print("\nVisual evaluation finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Visual evaluation of adaptation module in MuJoCo viewer",
    )
    parser.add_argument(
        "--adaptation-ckpt", required=True,
        help="Path to trained adaptation module checkpoint",
    )
    parser.add_argument(
        "--policy-ckpt", default=None,
        help="Path to adaptive Place policy checkpoint (optional)",
    )
    parser.add_argument("--robot", default="digit", choices=["digit", "g1", "h1"])
    parser.add_argument(
        "--dynamics-randomization", action="store_true",
        help="Enable DR so each reset gives different box properties",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_visual_eval(
        adaptation_ckpt=args.adaptation_ckpt,
        policy_ckpt=args.policy_ckpt,
        robot=args.robot,
        dynamics_randomization=args.dynamics_randomization,
        device=args.device,
    )


if __name__ == "__main__":
    main()
