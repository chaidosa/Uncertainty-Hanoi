"""Phase 2: PPO fine-tuning of the adaptive Place policy.

The existing Place policy (LSTM, 2-layer 64-D hidden) is extended with
9 extra input dimensions — the adaptation module's z_t (8-D) and
sigma_t (1-D) — and trained with PPO in MuJoCo while the adaptation
module runs in the loop but with **frozen** weights.

Usage
-----
    python -m adaptation.train_adaptive_place \
        --adaptation-ckpt checkpoints/adaptation_module.pt \
        --place-ckpt checkpoints/place_policy.pt \
        --robot digit \
        --total-steps 500_000_000 \
        --device cuda \
        --wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptation.adaptation_module import AdaptationModule, DEFAULT_LATENT_DIM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hyperparameters (match existing Place training where possible)
# ---------------------------------------------------------------------------

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
LR = 3e-4
NUM_ENVS = 64
STEPS_PER_ROLLOUT = 256
MINIBATCH_SIZE = 4096
PPO_EPOCHS = 4

# Curriculum: timestep thresholds
CURRICULUM_START = 100_000_000
CURRICULUM_END = 150_000_000

# Warmup prefix length when initialising GRU hidden state
GRU_WARMUP_STEPS = 80


# ---------------------------------------------------------------------------
# Policy and value networks
# ---------------------------------------------------------------------------

class AdaptivePlacePolicy(nn.Module):
    """LSTM policy for the adaptive Place skill.

    Observation layout:
      original_place_obs (obs_dim) | z_t (latent_dim) | sigma_t (1)
    """

    def __init__(
        self,
        obs_dim: int = 80,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = 20,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.ext_dim = latent_dim + 1
        self.total_dim = obs_dim + self.ext_dim
        self.hidden_dim = hidden_dim

        self.input_mlp = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.ELU(),
            nn.Linear(256, hidden_dim),
            nn.ELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = self.input_mlp(obs)
        x, hx = self.lstm(x, hx)
        mean = self.mean_head(x.squeeze(1))
        return mean, self.log_std.expand_as(mean), hx

    def get_action(self, obs, hx=None, deterministic=False):
        mean, log_std, hx = self.forward(obs, hx)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, hx

    def evaluate_actions(self, obs, actions, hx=None):
        mean, log_std, hx = self.forward(obs, hx)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, hx

    @classmethod
    def from_pretrained(cls, path: str, obs_dim: int = 80, action_dim: int = 20, **kwargs):
        """Load pre-trained Place weights and expand the input layer."""
        model = cls(obs_dim=obs_dim, action_dim=action_dim, **kwargs)
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            old_w = ckpt.get("input_mlp.0.weight")
            if old_w is not None and old_w.shape[1] == obs_dim:
                new_w = torch.zeros(old_w.shape[0], model.total_dim)
                new_w[:, :obs_dim] = old_w
                nn.init.normal_(new_w[:, obs_dim:], std=1e-4)
                ckpt["input_mlp.0.weight"] = new_w
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained Place weights from {path} "
                  f"(input expanded {obs_dim} -> {model.total_dim})")
        return model


class AdaptivePlaceCritic(nn.Module):
    """Privileged value function that additionally sees ground-truth box properties."""

    def __init__(self, obs_dim: int = 89, priv_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + priv_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, priv], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions from parallel environments for on-policy PPO."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, priv_dim: int):
        self.obs = torch.zeros(num_steps, num_envs, obs_dim)
        self.actions = torch.zeros(num_steps, num_envs, action_dim)
        self.log_probs = torch.zeros(num_steps, num_envs)
        self.rewards = torch.zeros(num_steps, num_envs)
        self.dones = torch.zeros(num_steps, num_envs)
        self.values = torch.zeros(num_steps, num_envs)
        self.priv = torch.zeros(num_steps, num_envs, priv_dim)
        self.sigmas = torch.zeros(num_steps, num_envs)
        self.returns = torch.zeros(num_steps, num_envs)
        self.advantages = torch.zeros(num_steps, num_envs)
        self.ptr = 0
        self.num_steps = num_steps
        self.num_envs = num_envs

    def reset(self):
        self.ptr = 0

    def insert(self, obs, action, log_prob, reward, done, value, priv, sigma=None):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.priv[self.ptr] = priv
        if sigma is not None:
            self.sigmas[self.ptr] = sigma
        self.ptr += 1

    def compute_returns_and_advantages(self, next_value: torch.Tensor):
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            delta = self.rewards[t] + GAMMA * next_val * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[t]) * last_gae
        self.returns = advantages + self.values
        self.advantages = advantages

    def get_batches(self, minibatch_size: int):
        total = self.num_steps * self.num_envs
        indices = torch.randperm(total)
        flat = lambda x: x.reshape(total, *x.shape[2:])  # noqa: E731
        for start in range(0, total, minibatch_size):
            idx = indices[start:start + minibatch_size]
            yield (
                flat(self.obs)[idx],
                flat(self.actions)[idx],
                flat(self.log_probs)[idx],
                flat(self.returns)[idx],
                flat(self.advantages)[idx],
                flat(self.priv)[idx],
            )

    def summary_stats(self) -> dict:
        """Return scalar summaries of the current rollout for logging."""
        return {
            "rollout/reward_mean": self.rewards.mean().item(),
            "rollout/reward_std": self.rewards.std().item(),
            "rollout/reward_min": self.rewards.min().item(),
            "rollout/reward_max": self.rewards.max().item(),
            "rollout/done_frac": self.dones.mean().item(),
            "rollout/value_mean": self.values.mean().item(),
            "rollout/sigma_mean": self.sigmas.mean().item(),
            "rollout/sigma_std": self.sigmas.std().item(),
            "rollout/advantage_mean": self.advantages.mean().item(),
        }


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

def curriculum_weight(total_steps_so_far: int, target_weight: float) -> float:
    """Linearly ramp ``target_weight`` from 0 between CURRICULUM_START and
    CURRICULUM_END global steps."""
    if total_steps_so_far < CURRICULUM_START:
        return 0.0
    if total_steps_so_far >= CURRICULUM_END:
        return target_weight
    frac = (total_steps_so_far - CURRICULUM_START) / (CURRICULUM_END - CURRICULUM_START)
    return target_weight * frac


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    adaptation_ckpt: str,
    place_ckpt: str | None = None,
    robot: str = "digit",
    total_steps: int = 500_000_000,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    use_wandb: bool = False,
    wandb_project: str = "humanoid-hanoi-adaptation",
    wandb_run_name: str | None = None,
    log_interval: int = 10,
    save_interval: int = 500,
):
    """Phase 2 PPO training loop.

    NOTE: The codebase ships environments but not vectorised parallel
    wrappers or learned Place checkpoints.  The rollout collection
    section below is marked with comments showing exactly where env
    calls go.  The PPO update itself is fully functional.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------------
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"phase2-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "phase": 2,
                "adaptation_ckpt": adaptation_ckpt,
                "place_ckpt": place_ckpt,
                "robot": robot,
                "total_steps": total_steps,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_ratio": CLIP_RATIO,
                "entropy_coef": ENTROPY_COEF,
                "vf_coef": VF_COEF,
                "lr": LR,
                "num_envs": NUM_ENVS,
                "steps_per_rollout": STEPS_PER_ROLLOUT,
                "minibatch_size": MINIBATCH_SIZE,
                "ppo_epochs": PPO_EPOCHS,
                "curriculum_start": CURRICULUM_START,
                "curriculum_end": CURRICULUM_END,
                "device": device,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("WARNING: --wandb requested but wandb is not installed. Continuing without it.")
        use_wandb = False

    # ------------------------------------------------------------------
    # 1. Load frozen adaptation module
    # ------------------------------------------------------------------
    adapt = AdaptationModule(device=device)
    adapt.load(adaptation_ckpt)
    adapt.eval()
    for p in adapt.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen adaptation module from {adaptation_ckpt}")

    # ------------------------------------------------------------------
    # 2. Build policy and critic
    # ------------------------------------------------------------------
    obs_dim = 80
    action_dim = 20
    priv_dim = 8
    extended_obs_dim = obs_dim + DEFAULT_LATENT_DIM + 1  # 89

    policy = AdaptivePlacePolicy.from_pretrained(
        place_ckpt, obs_dim=obs_dim, action_dim=action_dim,
    ).to(device)
    critic = AdaptivePlaceCritic(
        obs_dim=extended_obs_dim, priv_dim=priv_dim,
    ).to(device)

    all_params = list(policy.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(all_params, lr=LR)

    if use_wandb:
        wandb.watch(policy, log="gradients", log_freq=100)

    buf = RolloutBuffer(
        num_steps=STEPS_PER_ROLLOUT,
        num_envs=NUM_ENVS,
        obs_dim=extended_obs_dim,
        action_dim=action_dim,
        priv_dim=priv_dim,
    )

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    global_step = 0
    num_updates = total_steps // (STEPS_PER_ROLLOUT * NUM_ENVS)
    steps_per_update = STEPS_PER_ROLLOUT * NUM_ENVS

    print(f"Phase 2 training: {num_updates} updates "
          f"({STEPS_PER_ROLLOUT} x {NUM_ENVS} = {steps_per_update} steps/update)")
    print("NOTE: Plug in vectorised environments and a Place checkpoint to run end-to-end.\n")

    t_start = time.time()

    for update in range(1, num_updates + 1):
        update_start = time.time()

        # --- Linear LR decay ---
        frac = 1.0 - (update - 1) / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = LR * frac
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Curriculum weights ---
        w_efficiency = curriculum_weight(global_step, 0.02)
        w_fast_bonus = curriculum_weight(global_step, 0.5)

        # ---------------------------------------------------------------
        # --- Collect rollouts (env interaction) ---
        # ---------------------------------------------------------------
        # For each of STEPS_PER_ROLLOUT steps across NUM_ENVS:
        #   1. obs_80 = env.get_state()                          # (80,)
        #   2. adapt_obs = env.get_adaptation_obs()              # (59,)
        #   3. z_t, sigma_t = adapt.step(adapt_obs)              # (8,), (1,)
        #   4. ext_obs = cat(obs_80, z_t, sigma_t)               # (89,)
        #   5. action, log_prob, hx = policy.get_action(ext_obs) # (20,)
        #   6. env.step(action)
        #   7. reward = env.reward; done = env.compute_done()
        #   8. priv = env.get_privileged_box_properties()        # (8,)
        #   9. value = critic(ext_obs, priv)
        #  10. buf.insert(ext_obs, action, log_prob, reward, done, value, priv, sigma_t)
        #  11. On done: adapt.reset(); hx = None; env.reset()
        #
        # Apply curriculum: scale r_efficiency by w_efficiency,
        #                    scale r_fast_bonus by w_fast_bonus.
        # ---------------------------------------------------------------

        buf.reset()
        global_step += steps_per_update

        # --- Compute advantages (after rollout) ---
        # next_value = critic(last_ext_obs, last_priv)
        # buf.compute_returns_and_advantages(next_value)

        # ---------------------------------------------------------------
        # --- PPO update ---
        # ---------------------------------------------------------------
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_clip_frac = 0.0
        epoch_approx_kl = 0.0
        n_minibatches = 0

        for _ppo_epoch in range(PPO_EPOCHS):
            for (mb_obs, mb_act, mb_old_lp, mb_ret, mb_adv, mb_priv) in \
                    buf.get_batches(MINIBATCH_SIZE):

                mb_obs = mb_obs.to(device)
                mb_act = mb_act.to(device)
                mb_old_lp = mb_old_lp.to(device)
                mb_ret = mb_ret.to(device)
                mb_adv = mb_adv.to(device)
                mb_priv = mb_priv.to(device)

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_lp, entropy, _ = policy.evaluate_actions(mb_obs, mb_act)
                ratio = (new_lp - mb_old_lp).exp()
                log_ratio = new_lp - mb_old_lp

                # Clipped surrogate
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - CLIP_RATIO, 1 + CLIP_RATIO) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = critic(mb_obs, mb_priv)
                value_loss = 0.5 * (mb_ret - value_pred).pow(2).mean()

                # Entropy
                ent_mean = entropy.mean()

                loss = policy_loss + VF_COEF * value_loss - ENTROPY_COEF * ent_mean

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(all_params, MAX_GRAD_NORM)
                optimizer.step()

                # Tracking
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > CLIP_RATIO).float().mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += ent_mean.item()
                epoch_clip_frac += clip_frac.item()
                epoch_approx_kl += approx_kl.item()
                n_minibatches += 1

        # --- Per-update averages ---
        n_mb = max(n_minibatches, 1)
        update_time = time.time() - update_start
        sps = steps_per_update / update_time if update_time > 0 else 0

        metrics = {
            "global_step": global_step,
            "update": update,
            "ppo/policy_loss": epoch_policy_loss / n_mb,
            "ppo/value_loss": epoch_value_loss / n_mb,
            "ppo/entropy": epoch_entropy / n_mb,
            "ppo/clip_frac": epoch_clip_frac / n_mb,
            "ppo/approx_kl": epoch_approx_kl / n_mb,
            "ppo/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": current_lr,
            "curriculum/w_efficiency": w_efficiency,
            "curriculum/w_fast_bonus": w_fast_bonus,
            "perf/update_time_s": update_time,
            "perf/steps_per_sec": sps,
        }
        metrics.update(buf.summary_stats())

        # --- Console logging ---
        if update % log_interval == 0 or update == 1:
            elapsed = time.time() - t_start
            print(
                f"Update {update:>6d}/{num_updates} | "
                f"step {global_step:>12,} | "
                f"pi_loss {metrics['ppo/policy_loss']:+.4f} | "
                f"v_loss {metrics['ppo/value_loss']:.4f} | "
                f"ent {metrics['ppo/entropy']:.4f} | "
                f"clip {metrics['ppo/clip_frac']:.3f} | "
                f"kl {metrics['ppo/approx_kl']:.4f} | "
                f"rew {metrics['rollout/reward_mean']:.3f} | "
                f"lr {current_lr:.2e} | "
                f"SPS {sps:.0f} | "
                f"{elapsed / 60:.1f}m"
            )

        # --- W&B logging ---
        if use_wandb:
            wandb.log(metrics, step=global_step)

        # --- Checkpoint ---
        if update % save_interval == 0:
            ckpt_path_p = os.path.join(save_dir, f"adaptive_place_policy_{global_step}.pt")
            ckpt_path_c = os.path.join(save_dir, f"adaptive_place_critic_{global_step}.pt")
            torch.save(policy.state_dict(), ckpt_path_p)
            torch.save(critic.state_dict(), ckpt_path_c)
            if use_wandb:
                artifact = wandb.Artifact(
                    f"adaptive-place-{global_step}", type="model",
                    metadata={"global_step": global_step, "update": update},
                )
                artifact.add_file(ckpt_path_p)
                artifact.add_file(ckpt_path_c)
                wandb.log_artifact(artifact)
            print(f"  -> Saved checkpoint at step {global_step}")

    # --- Final save ---
    final_p = os.path.join(save_dir, "adaptive_place_policy_final.pt")
    final_c = os.path.join(save_dir, "adaptive_place_critic_final.pt")
    torch.save(policy.state_dict(), final_p)
    torch.save(critic.state_dict(), final_c)

    total_time = time.time() - t_start
    print(f"\nPhase 2 training complete in {total_time / 3600:.1f} h.")

    if use_wandb:
        wandb.run.summary["total_train_time_h"] = total_time / 3600
        artifact = wandb.Artifact("adaptive-place-final", type="model")
        artifact.add_file(final_p)
        artifact.add_file(final_c)
        wandb.log_artifact(artifact)
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train adaptive Place policy")
    parser.add_argument("--adaptation-ckpt", required=True,
                        help="Path to trained adaptation module checkpoint")
    parser.add_argument("--place-ckpt", default=None,
                        help="Path to pretrained Place policy checkpoint (optional)")
    parser.add_argument("--robot", default="digit", choices=["digit", "g1", "h1"])
    parser.add_argument("--total-steps", type=int, default=500_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="humanoid-hanoi-adaptation")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    args = parser.parse_args()

    train(
        adaptation_ckpt=args.adaptation_ckpt,
        place_ckpt=args.place_ckpt,
        robot=args.robot,
        total_steps=args.total_steps,
        device=args.device,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
