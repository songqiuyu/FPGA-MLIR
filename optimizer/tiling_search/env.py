"""
Tiling Search Environment (Phase 3-A)
Gym-compatible environment that wraps the buffer constraint checker
and rewards the RL agent for finding tight-but-legal (tM, tR, tC) tiles.

State:  (N, M, R, C, k, s, d) - layer parameters, normalized to [0,1]
Action: (delta_tM, delta_tR, delta_tC) - discrete reduction/expansion steps
Reward: Buffer utilization score if legal, -1 penalty if any limit exceeded.
"""

import math
import numpy as np

from coa.tiling import (
    calculate_buffer_consumption,
    get_tile,
    buffer_utilization as _coa_buffer_utilization,
    WDEPTH_LIMIT,
    GDEPTH_LIMIT,
    ODEPTH_LIMIT,
)

# ---- Tile step candidates for each dimension ----
_STEP_VALUES = [1, 2, 4, 8, 16, 32, 64]


def buffer_utilization(tN, tM, tR, tC, N, M, k, s, d):
    """Return (wutil, gutil, outil) in [0, 1]; > 1.0 means over-limit."""
    return _coa_buffer_utilization(tN, tM, tR, tC, N, M, k, s, d)


class TilingEnv:
    """
    Discrete tiling search environment.

    Actions (9 total):
      0: tM /= 2    1: tM *= 2    2: no-op tM
      3: tR /= 2    4: tR *= 2    5: no-op tR
      6: tC /= 2    7: tC *= 2    8: done (submit current tile)

    State (7-dim float vector):
      [N/512, M/512, R/512, C/512, k/7, s/4, d/4,
       wUtil, gUtil, oUtil, tM/M, tR/R, tC/C]  -> 13-dim
    """

    N_ACTIONS = 9
    STATE_DIM = 13
    MAX_STEPS = 50

    def __init__(self, N: int, M: int, R: int, C: int,
                 k: int = 3, s: int = 1, d: int = 1):
        self.N = N
        self.M = M
        self.R = R
        self.C = C
        self.k = k
        self.s = s
        self.d = d
        self._reset_tile()

    def _reset_tile(self):
        self.tM = self.M
        self.tR = self.R
        self.tC = self.C
        self.steps = 0
        self.done = False

    def reset(self):
        self._reset_tile()
        return self._obs()

    def _obs(self):
        wU, gU, oU = buffer_utilization(
            self.N, self.tM, self.tR, self.tC,
            self.N, self.M, self.k, self.s, self.d)
        return np.array([
            self.N / 512.0, self.M / 512.0,
            self.R / 512.0, self.C / 512.0,
            self.k / 7.0,  self.s / 4.0, self.d / 4.0,
            wU, gU, oU,
            self.tM / max(self.M, 1),
            self.tR / max(self.R, 1),
            self.tC / max(self.C, 1),
        ], dtype=np.float32)

    def _is_legal(self, tM=None, tR=None, tC=None):
        tM = tM or self.tM
        tR = tR or self.tR
        tC = tC or self.tC
        flag = calculate_buffer_consumption(
            self.N, tM, tR, tC,
            self.N, self.M, self.k, self.s, 0, self.d)
        return flag == 0

    def _utilization_reward(self):
        wU, gU, oU = buffer_utilization(
            self.N, self.tM, self.tR, self.tC,
            self.N, self.M, self.k, self.s, self.d)
        if max(wU, gU, oU) > 1.0:
            return -1.0
        # Reward = mean utilization (higher = better use of hardware)
        return (wU + gU + oU) / 3.0

    def step(self, action: int):
        assert 0 <= action < self.N_ACTIONS, f"Invalid action {action}"
        self.steps += 1

        prev_tM, prev_tR, prev_tC = self.tM, self.tR, self.tC

        if action == 0:   self.tM = max(16, self.tM // 2)
        elif action == 1: self.tM = min(self.M, self.tM * 2)
        elif action == 3: self.tR = max(1, self.tR // 2)
        elif action == 4: self.tR = min(self.R, self.tR * 2)
        elif action == 6: self.tC = max(1, self.tC // 2)
        elif action == 7: self.tC = min(self.C, self.tC * 2)
        elif action == 8:
            # Submit - compute final reward
            reward = self._utilization_reward()
            self.done = True
            return self._obs(), reward, True, {}

        reward = self._utilization_reward()
        done = self.steps >= self.MAX_STEPS
        self.done = done
        return self._obs(), reward, done, {}

    @property
    def best_tile(self):
        return (self.tM, self.tR, self.tC)

    @classmethod
    def from_coa_layer(cls, layer_params: dict):
        """Create env from a layer parameter dict (as used in extract_vliw.py)."""
        return cls(
            N=layer_params.get('N', 64),
            M=layer_params.get('M', 64),
            R=layer_params.get('R', 56),
            C=layer_params.get('C', 56),
            k=layer_params.get('kernel', [3, 3])[0],
            s=layer_params.get('strides', [1, 1])[0],
            d=layer_params.get('dilations', [1, 1])[0],
        )
