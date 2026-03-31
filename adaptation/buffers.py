"""Rolling observation buffer for the GRU-based adaptation module.

Maintains a fixed-size FIFO window of recent observations and provides
batched retrieval for sequence processing.
"""

import torch
import numpy as np


class RollingObservationBuffer:
    """Fixed-size circular buffer storing the most recent observations.

    The buffer stores `window_size` observations of dimension `obs_dim`.
    Before the buffer is full, older slots are zero-padded.  A validity
    mask is maintained so downstream consumers can distinguish real
    observations from padding.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of each observation vector.
    window_size : int
        Number of most-recent timesteps to keep (default 50, i.e. 1 s at 50 Hz).
    device : str
        Torch device for the internal tensor storage.
    """

    def __init__(self, obs_dim: int, window_size: int = 50, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.window_size = window_size
        self.device = device
        self.reset()

    def reset(self):
        """Clear the buffer (e.g. when starting interaction with a new box)."""
        self._buffer = torch.zeros(self.window_size, self.obs_dim, device=self.device)
        self._valid = torch.zeros(self.window_size, dtype=torch.bool, device=self.device)
        self._ptr = 0
        self._count = 0

    def push(self, obs: np.ndarray | torch.Tensor):
        """Append a single observation to the buffer.

        Parameters
        ----------
        obs : array-like, shape (obs_dim,)
            The observation vector for the current timestep.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        self._buffer[self._ptr] = obs
        self._valid[self._ptr] = True
        self._ptr = (self._ptr + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)

    def get_sequence(self) -> torch.Tensor:
        """Return the buffered observations in chronological order.

        Returns
        -------
        torch.Tensor, shape (window_size, obs_dim)
            Oldest observation first, zero-padded if fewer than
            `window_size` observations have been pushed.
        """
        if self._count < self.window_size:
            return self._buffer.clone()
        # Reorder so that the oldest observation comes first
        idx = torch.arange(self.window_size, device=self.device)
        idx = (idx + self._ptr) % self.window_size
        return self._buffer[idx]

    def get_valid_mask(self) -> torch.Tensor:
        """Return a boolean mask aligned with `get_sequence` output."""
        if self._count < self.window_size:
            return self._valid.clone()
        idx = torch.arange(self.window_size, device=self.device)
        idx = (idx + self._ptr) % self.window_size
        return self._valid[idx]

    @property
    def num_valid(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count
