"""GRU-based adaptation module for online box-property estimation.

Processes proprioceptive + contact history to produce:
  - z_t  (latent_dim-D)  : latent embedding of estimated physical box properties
  - sigma_t (scalar)     : mean variance across latent dims (uncertainty)

The GRU hidden state persists across Pickup -> GoTo-with-Box -> Place for
the same box, and is reset when a new box interaction begins.

Architecture
------------
  Input MLP : Linear(input_dim, 128) -> ELU -> Linear(128, 64)
  GRU       : GRU(input_size=64, hidden_size=64, num_layers=1)
  Mean head : Linear(64, latent_dim)
  LogVar head : Linear(64, latent_dim)

Digit V3 observation breakdown (input_dim = 59)
------------------------------------------------
  base_orient           4   (quaternion wxyz)
  base_ang_vel          3   (IMU gyro)
  motor_pos            20   (actuated joints)
  motor_vel            20   (actuated joint velocities)
  hand_contact_force    6   (left + right elbow contact, 3D each)
  box_pose_rel          6   (xyz + rpy of current box in robot base frame)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from adaptation.buffers import RollingObservationBuffer


# ---------------------------------------------------------------------------
# Default dimensions matching the Digit V3 observation layout
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIM = 59
DEFAULT_LATENT_DIM = 8
DEFAULT_HIDDEN_DIM = 64
DEFAULT_WINDOW_SIZE = 50  # 1 second at 50 Hz


class AdaptationModule(nn.Module):
    """GRU encoder that estimates latent box properties from interaction history.

    Parameters
    ----------
    input_dim : int
        Per-timestep observation dimensionality.
    latent_dim : int
        Size of the output latent vector z_t.
    hidden_dim : int
        GRU (and intermediate MLP) hidden size.
    window_size : int
        Rolling buffer length for the input MLP/GRU.
    device : str
        Torch device.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        window_size: int = DEFAULT_WINDOW_SIZE,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.device = device

        # Input MLP
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, hidden_dim),
        )

        # Temporal encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
        )

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Persistent GRU hidden state (carries across skill boundaries for
        # the same box; reset explicitly between different boxes).
        self.hidden: torch.Tensor | None = None

        # Per-episode rolling observation buffer
        self.obs_buffer = RollingObservationBuffer(
            obs_dim=input_dim, window_size=window_size, device=device,
        )

        self.to(device)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self):
        """Reset hidden state and buffer (call when starting a new box)."""
        self.hidden = torch.zeros(1, 1, self.hidden_dim, device=self.device)
        self.obs_buffer.reset()

    # ------------------------------------------------------------------
    # Forward pass variants
    # ------------------------------------------------------------------

    def step(self, obs_t: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Online inference: process a single timestep observation.

        Pushes the observation into the rolling buffer, runs through the
        input MLP, and advances the GRU by one step (using the persistent
        hidden state).

        Returns (z_t, sigma_t) — both detached from the computation graph
        when used during Place policy rollouts with a frozen adaptation
        module.

        Parameters
        ----------
        obs_t : array-like, shape (input_dim,)

        Returns
        -------
        z_t : torch.Tensor, shape (latent_dim,)
        sigma_t : torch.Tensor, shape (1,)
        """
        if self.hidden is None:
            self.reset()

        if isinstance(obs_t, np.ndarray):
            obs_t = torch.from_numpy(obs_t).float().to(self.device)

        self.obs_buffer.push(obs_t)

        # Input MLP: (input_dim,) -> (hidden_dim,)
        x = self.input_mlp(obs_t)
        # GRU expects (seq_len=1, batch=1, hidden_dim)
        x = x.unsqueeze(0).unsqueeze(0)
        output, self.hidden = self.gru(x, self.hidden)

        # Decode
        h = output.squeeze(0).squeeze(0)  # (hidden_dim,)
        z_mean = self.mean_head(h)
        z_logvar = self.logvar_head(h)

        z_t = z_mean
        sigma_t = torch.mean(torch.exp(z_logvar)).unsqueeze(0)

        return z_t, sigma_t

    def forward_sequence(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch forward over a full observation sequence (for training).

        Parameters
        ----------
        obs_seq : torch.Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        z_means : torch.Tensor, shape (batch, seq_len, latent_dim)
        z_logvars : torch.Tensor, shape (batch, seq_len, latent_dim)
        """
        batch, seq_len, _ = obs_seq.shape
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        x = self.input_mlp(obs_seq)
        # GRU expects (seq_len, batch, hidden_dim)
        x = x.transpose(0, 1)
        h0 = torch.zeros(1, batch, self.hidden_dim, device=obs_seq.device)
        output, _ = self.gru(x, h0)
        # output: (seq_len, batch, hidden_dim)
        output = output.transpose(0, 1)  # (batch, seq_len, hidden_dim)

        z_means = self.mean_head(output)
        z_logvars = self.logvar_head(output)

        return z_means, z_logvars

    # ------------------------------------------------------------------
    # Uncertainty summary
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sigma(z_logvar: torch.Tensor) -> torch.Tensor:
        """Compute scalar uncertainty from per-dimension log-variance.

        Parameters
        ----------
        z_logvar : torch.Tensor, shape (..., latent_dim)

        Returns
        -------
        sigma : torch.Tensor, shape (..., 1)
        """
        return torch.mean(torch.exp(z_logvar), dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
