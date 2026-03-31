"""Phase 1: Supervised (privileged) training of the GRU adaptation module.

The adaptation module is trained to regress ground-truth box physical
properties from proprioceptive + contact observation history via a
combination of MSE loss and a KL regulariser on the variance head.

Loss
----
    L = MSE(z_t, labels) + beta * KL( N(mu, sigma) || N(0, I) )

where beta = 0.01 keeps the variance head from collapsing to zero.

Usage
-----
    python -m adaptation.train_adaptation \
        --data data/adaptation_rollouts.npz \
        --epochs 300 \
        --batch-size 256 \
        --lr 3e-4 \
        --output checkpoints/adaptation_module.pt \
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
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptation.adaptation_module import AdaptationModule, DEFAULT_INPUT_DIM, DEFAULT_LATENT_DIM

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdaptationDataset(Dataset):
    """Variable-length rollout sequences padded to a fixed max length."""

    def __init__(self, npz_path: str, max_len: int = 300):
        data = np.load(npz_path, allow_pickle=True)
        self.obs_seqs = data["obs_seqs"]
        self.label_seqs = data["label_seqs"]
        self.lengths = data["lengths"].astype(np.int64)
        self.max_len = max_len

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        L = int(self.lengths[idx])
        obs = np.zeros((self.max_len, DEFAULT_INPUT_DIM), dtype=np.float32)
        lab = np.zeros((self.max_len, DEFAULT_LATENT_DIM), dtype=np.float32)
        obs[:L] = self.obs_seqs[idx][:L]
        lab[:L] = self.label_seqs[idx][:L]
        return (
            torch.from_numpy(obs),
            torch.from_numpy(lab),
            torch.tensor(L, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """MSE loss averaged only over valid timesteps."""
    B, T, D = pred.shape
    mask = torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum()


def masked_per_dim_mse(
    pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Per-dimension MSE over valid timesteps.  Returns shape (D,)."""
    B, T, D = pred.shape
    mask = torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target) ** 2
    return (diff * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1))


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma) || N(0, I) ) averaged over valid timesteps."""
    B, T, D = mu.shape
    mask = torch.arange(T, device=mu.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(mu)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return (kl * mask).sum() / mask.sum()


def mean_sigma(logvar: torch.Tensor, lengths: torch.Tensor) -> float:
    """Mean sigma across valid final timesteps."""
    B, T, D = logvar.shape
    sigma = torch.mean(torch.exp(logvar), dim=-1)  # (B, T)
    final_sigma = []
    for b in range(B):
        L = int(lengths[b])
        if L > 0:
            final_sigma.append(sigma[b, L - 1].item())
    return np.mean(final_sigma) if final_sigma else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

LABEL_NAMES = ["mass", "friction", "com_x", "com_y", "com_z", "size_x", "size_y", "size_z"]


def train(
    data_path: str,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 3e-4,
    beta: float = 0.01,
    val_split: float = 0.1,
    output_path: str = "checkpoints/adaptation_module.pt",
    device: str = "cpu",
    use_wandb: bool = False,
    wandb_project: str = "humanoid-hanoi-adaptation",
    wandb_run_name: str | None = None,
):
    # ------------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------------
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"phase1-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "phase": 1,
                "data_path": data_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "beta": beta,
                "val_split": val_split,
                "input_dim": DEFAULT_INPUT_DIM,
                "latent_dim": DEFAULT_LATENT_DIM,
                "device": device,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("WARNING: --wandb requested but wandb is not installed. Continuing without it.")
        use_wandb = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = AdaptationDataset(data_path)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"Dataset: {len(dataset)} episodes  |  train {n_train}  |  val {n_val}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AdaptationModule(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=50)

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_kl_sum = 0.0
        train_steps = 0
        grad_norm = 0.0
        for obs, labels, lengths in train_loader:
            obs, labels, lengths = obs.to(device), labels.to(device), lengths.to(device)
            z_means, z_logvars = model.forward_sequence(obs)

            loss_mse = masked_mse(z_means, labels, lengths)
            loss_kl = kl_divergence(z_means, z_logvars, lengths)
            loss = loss_mse + beta * loss_kl

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_mse_sum += loss_mse.item()
            train_kl_sum += loss_kl.item()
            train_steps += 1

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_kl_sum = 0.0
        val_sigma_sum = 0.0
        val_per_dim_mse = torch.zeros(DEFAULT_LATENT_DIM, device=device)
        val_steps = 0
        with torch.no_grad():
            for obs, labels, lengths in val_loader:
                obs, labels, lengths = obs.to(device), labels.to(device), lengths.to(device)
                z_means, z_logvars = model.forward_sequence(obs)
                loss_mse = masked_mse(z_means, labels, lengths)
                loss_kl = kl_divergence(z_means, z_logvars, lengths)
                val_loss_sum += (loss_mse + beta * loss_kl).item()
                val_mse_sum += loss_mse.item()
                val_kl_sum += loss_kl.item()
                val_per_dim_mse += masked_per_dim_mse(z_means, labels, lengths)
                val_sigma_sum += mean_sigma(z_logvars, lengths)
                val_steps += 1

        avg_train = train_loss_sum / max(train_steps, 1)
        avg_train_mse = train_mse_sum / max(train_steps, 1)
        avg_train_kl = train_kl_sum / max(train_steps, 1)
        avg_val = val_loss_sum / max(val_steps, 1)
        avg_val_mse = val_mse_sum / max(val_steps, 1)
        avg_val_kl = val_kl_sum / max(val_steps, 1)
        avg_val_sigma = val_sigma_sum / max(val_steps, 1)
        avg_val_per_dim = (val_per_dim_mse / max(val_steps, 1)).cpu().numpy()
        current_lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - ep_start

        # --- Console logging ---
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train {avg_train:.5f} (mse {avg_train_mse:.5f} kl {avg_train_kl:.5f}) | "
                f"val {avg_val:.5f} (mse {avg_val_mse:.5f} kl {avg_val_kl:.5f}) | "
                f"sigma {avg_val_sigma:.4f} | lr {current_lr:.2e} | {ep_time:.1f}s"
            )

        # --- W&B logging ---
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train,
                "train/mse": avg_train_mse,
                "train/kl": avg_train_kl,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "val/loss": avg_val,
                "val/mse": avg_val_mse,
                "val/kl": avg_val_kl,
                "val/sigma_mean": avg_val_sigma,
                "lr": current_lr,
                "epoch_time_s": ep_time,
            }
            for i, name in enumerate(LABEL_NAMES):
                log_dict[f"val/mse_{name}"] = float(avg_val_per_dim[i])
            wandb.log(log_dict, step=epoch)

        # --- Best model checkpoint ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            model.save(output_path)
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time / 60:.1f} min.  Best val loss: {best_val_loss:.5f}")
    print(f"Saved best model to {output_path}")

    if use_wandb:
        wandb.run.summary["total_train_time_min"] = total_time / 60
        artifact = wandb.Artifact("adaptation-module", type="model")
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)
        wandb.finish()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_ood(
    model: AdaptationModule,
    dataset: AdaptationDataset,
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Check that sigma_t is higher for out-of-distribution samples.

    Returns a dict of results that can be logged to W&B.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_sigma: list[float] = []
    all_mass: list[float] = []
    all_z: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for obs, labels, lengths in loader:
            obs, labels, lengths = obs.to(device), labels.to(device), lengths.to(device)
            z_means, z_logvars = model.forward_sequence(obs)
            sigma = AdaptationModule.compute_sigma(z_logvars)
            for b in range(obs.shape[0]):
                L = int(lengths[b])
                if L > 0:
                    all_sigma.append(sigma[b, L - 1, 0].item())
                    all_mass.append(labels[b, 0, 0].item())
                    all_z.append(z_means[b, L - 1].cpu().numpy())
                    all_labels.append(labels[b, 0].cpu().numpy())

    masses = np.array(all_mass)
    sigmas = np.array(all_sigma)
    z_preds = np.stack(all_z)
    gt_labels = np.stack(all_labels)

    q25, q75 = np.percentile(masses, [25, 75])
    in_dist = (masses >= q25) & (masses <= q75)
    ood = ~in_dist

    results = {
        "ood/in_dist_sigma_mean": float(sigmas[in_dist].mean()),
        "ood/in_dist_sigma_std": float(sigmas[in_dist].std()),
        "ood/ood_sigma_mean": float(sigmas[ood].mean()),
        "ood/ood_sigma_std": float(sigmas[ood].std()),
        "ood/in_dist_count": int(in_dist.sum()),
        "ood/ood_count": int(ood.sum()),
        "ood/mass_q25": float(q25),
        "ood/mass_q75": float(q75),
    }

    # Per-dimension RMSE at final timestep
    per_dim_rmse = np.sqrt(np.mean((z_preds - gt_labels) ** 2, axis=0))
    for i, name in enumerate(LABEL_NAMES):
        results[f"ood/rmse_{name}"] = float(per_dim_rmse[i])
    results["ood/rmse_total"] = float(np.sqrt(np.mean((z_preds - gt_labels) ** 2)))

    print(f"In-distribution  mass [{q25:.2f}, {q75:.2f}]: "
          f"mean sigma = {results['ood/in_dist_sigma_mean']:.4f}  (n={results['ood/in_dist_count']})")
    print(f"Out-of-distribution: "
          f"mean sigma = {results['ood/ood_sigma_mean']:.4f}  (n={results['ood/ood_count']})")
    print(f"Total RMSE at final step: {results['ood/rmse_total']:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>10s} RMSE: {per_dim_rmse[i]:.4f}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(results)

        # Sigma vs mass scatter table
        table = wandb.Table(columns=["mass", "sigma", "distribution"])
        for m, s, is_id in zip(masses, sigmas, in_dist):
            table.add_data(float(m), float(s), "in-dist" if is_id else "OOD")
        wandb.log({"ood/sigma_vs_mass": wandb.plot.scatter(
            table, "mass", "sigma", title="Sigma vs Box Mass",
        )})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train adaptation module")
    parser.add_argument("--data", required=True, help="Path to .npz rollout data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output", default="checkpoints/adaptation_module.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="humanoid-hanoi-adaptation")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--eval-ood", action="store_true",
                        help="After training, run OOD sigma analysis")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        val_split=args.val_split,
        output_path=args.output,
        device=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    if args.eval_ood:
        print("\n--- OOD uncertainty analysis ---")
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=f"phase1-eval-{time.strftime('%Y%m%d-%H%M%S')}",
                config={"phase": "1-eval", "checkpoint": args.output},
            )
        model = AdaptationModule(device=args.device)
        model.load(args.output)
        ds = AdaptationDataset(args.data)
        evaluate_ood(model, ds, device=args.device, use_wandb=args.wandb)
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
