"""M2 Sensor Augmentation: Jitter, Scale, and 3D Sensor Rotation (Um et al. recipe)."""

from typing import List, Tuple, Optional
import numpy as np


def build_3d_rotation_matrix(angle_x_deg: float, angle_y_deg: float, angle_z_deg: float) -> np.ndarray:
    """Builds a 3x3 orthonormal rotation matrix from Euler angles in degrees."""
    ax, ay, az = np.radians([angle_x_deg, angle_y_deg, angle_z_deg])
    
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(ax), -np.sin(ax)],
        [0.0, np.sin(ax), np.cos(ax)]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(ay), 0.0, np.sin(ay)],
        [0.0, 1.0, 0.0],
        [-np.sin(ay), 0.0, np.cos(ay)]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az), np.cos(az), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx


class SensorAugmentor:
    """
    On-the-fly sensor augmentation (M2).
    Applied strictly to training windows.
    Transforms (each applied independently with probability p=0.5):
      1. Jitter: additive Gaussian noise ~ N(0, 0.05^2) in standardized units.
      2. Scaling: per-channel multiplicative scaling ~ N(1, 0.1^2).
      3. Rotation: 3D rotation angles ~ U(-30°, +30°) applied jointly to triaxial sensor pairs.
    """

    def __init__(
        self,
        p_each: float = 0.5,
        jitter_sigma: float = 0.05,
        scale_mean: float = 1.0,
        scale_std: float = 0.1,
        rotation_deg_max: float = 30.0,
        random_state: Optional[int] = None
    ):
        self.p_each = p_each
        self.jitter_sigma = jitter_sigma
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.rotation_deg_max = rotation_deg_max
        self.rng = np.random.RandomState(random_state)

    def set_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def apply_jitter(self, x: np.ndarray) -> np.ndarray:
        # x shape: (C, 128)
        noise = self.rng.normal(0.0, self.jitter_sigma, size=x.shape).astype(np.float32)
        return x + noise

    def apply_scaling(self, x: np.ndarray) -> np.ndarray:
        # x shape: (C, 128)
        C = x.shape[0]
        factors = self.rng.normal(self.scale_mean, self.scale_std, size=(C, 1)).astype(np.float32)
        return x * factors

    def apply_rotation(self, x: np.ndarray) -> np.ndarray:
        # x shape: (C, 128)
        C = x.shape[0]
        out = x.copy()
        
        # Triad blocks of 3 channels (x, y, z)
        # Accelerometer and gyroscope of the same sensor unit share the rotation matrix
        n_triads = C // 3
        triad_idx = 0
        while triad_idx < n_triads:
            # Draw rotation angles for this physical sensor unit
            angles = self.rng.uniform(-self.rotation_deg_max, self.rotation_deg_max, size=3)
            R = build_3d_rotation_matrix(angles[0], angles[1], angles[2])

            # Apply to acc triad
            start_ch = triad_idx * 3
            out[start_ch:start_ch + 3, :] = R @ out[start_ch:start_ch + 3, :]
            triad_idx += 1

            # If next triad is the coupled gyroscope for the same IMU, apply same R
            if triad_idx < n_triads:
                start_ch_gyro = triad_idx * 3
                out[start_ch_gyro:start_ch_gyro + 3, :] = R @ out[start_ch_gyro:start_ch_gyro + 3, :]
                triad_idx += 1

        return out

    def transform_window(self, x: np.ndarray) -> np.ndarray:
        """Transforms a single window (C, 128) with on-the-fly random draws."""
        res = x.copy()
        if self.rng.uniform() < self.p_each:
            res = self.apply_jitter(res)
        if self.rng.uniform() < self.p_each:
            res = self.apply_scaling(res)
        if self.rng.uniform() < self.p_each:
            res = self.apply_rotation(res)
        return res

    def transform_batch(self, batch: np.ndarray) -> np.ndarray:
        """Transforms a batch of windows (B, C, 128)."""
        B = batch.shape[0]
        out = np.zeros_like(batch)
        for b in range(B):
            out[b] = self.transform_window(batch[b])
        return out
