"""
Tiling Search - Bayesian Optimization (Phase 3-A, Method 1)
Uses optuna to find optimal (tM, tR, tC) for each layer by treating
calculate_buffer_consumption() as a black-box objective.

Output: A JSON lookup table that can be embedded into the --coa-tiling pass
        or used standalone to precompute tiles for known layer shapes.
"""

import json
import math
import sys
import os
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'old_version', 'tools'))
from assign_addr import calculate_buffer_consumption, get_tile

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    print("[bayesian_opt] optuna not found. Install: pip install optuna")

# ---- Buffer limits ----
WDEPTH_LIMIT = 256
GDEPTH_LIMIT = 1024
ODEPTH_LIMIT = 2048


def _score(tM: int, tR: int, tC: int,
           N: int, M: int, R: int, C: int,
           k: int, s: int, d: int) -> float:
    """
    Objective to MAXIMIZE: average buffer utilization (higher = better).
    Returns -inf if any constraint is violated.
    """
    flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, k, s, 0, d)
    if flag != 0:
        return -1.0

    # Compute utilization
    def ceil32(x): return math.ceil(x / 32)
    def ceil16(x): return math.ceil(x / 16)

    tN32 = max(ceil32(sn + tM) - sn // 32 for sn in range(0, N, tM)) if N > 0 else 1
    tM32 = max(ceil16(sm + tM) - sm // 16 for sm in range(0, M, tM)) if M > 0 else 1

    relems = (tR - 1) * s + (k - 1) * d + 1
    celems = (tC - 1) * s + (k - 1) * d + 1
    gdepth = relems * celems * tN32
    wdepth = tN32 * k * k * tM32
    odepth = tR * tC * tM32

    wU = wdepth / WDEPTH_LIMIT
    gU = gdepth / GDEPTH_LIMIT
    oU = odepth / ODEPTH_LIMIT

    return float((wU + gU + oU) / 3.0)


def bayesian_tile_search(N: int, M: int, R: int, C: int,
                          k: int = 3, s: int = 1, d: int = 1,
                          n_trials: int = 200) -> Tuple[int, int, int]:
    """
    Run Bayesian optimization to find the best (tM, tR, tC).
    Falls back to the greedy heuristic if optuna is unavailable.
    """
    if not _HAS_OPTUNA:
        return get_tile(N, M, R, C, k, s, 0, d)

    # Valid tM values: multiples of 16 up to M
    valid_tM = [v for v in range(16, M + 1, 16)] or [16]
    valid_tR = [v for v in range(1, R + 1)]
    valid_tC = [v for v in range(1, C + 1)]

    def objective(trial):
        tM = trial.suggest_categorical("tM", valid_tM)
        tR = trial.suggest_categorical("tR", valid_tR)
        tC = trial.suggest_categorical("tC", valid_tC)
        return _score(tM, tR, tC, N, M, R, C, k, s, d)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    tM, tR, tC = best["tM"], best["tR"], best["tC"]

    # Verify the result is actually legal
    if calculate_buffer_consumption(N, tM, tR, tC, N, M, k, s, 0, d) != 0:
        # Fallback to greedy
        return get_tile(N, M, R, C, k, s, 0, d)

    return tM, tR, tC


def build_tile_lookup(layer_list: List[Dict]) -> Dict:
    """
    Build a lookup table {layer_key: {tM, tR, tC, score}} for a list of layers.
    Each layer dict must contain: N, M, R, C, k, s, d.
    """
    table = {}
    for i, layer in enumerate(layer_list):
        N = layer.get("N", 64)
        M = layer.get("M", 64)
        R = layer.get("R", 56)
        C = layer.get("C", 56)
        k = layer.get("k", 3)
        s = layer.get("s", 1)
        d = layer.get("d", 1)
        n_trials = layer.get("n_trials", 200)

        # Use shape as key to deduplicate identical layer shapes
        key = f"N{N}_M{M}_R{R}_C{C}_k{k}_s{s}_d{d}"
        if key in table:
            print(f"  Layer {i}: cache hit for {key}")
            continue

        print(f"  Layer {i}: searching tiles for {key} ...")
        # Compare greedy vs Bayesian
        greedy_tM, greedy_tR, greedy_tC = get_tile(N, M, R, C, k, s, 0, d)
        greedy_score = _score(greedy_tM, greedy_tR, greedy_tC, N, M, R, C, k, s, d)

        bayes_tM, bayes_tR, bayes_tC = bayesian_tile_search(
            N, M, R, C, k, s, d, n_trials)
        bayes_score = _score(bayes_tM, bayes_tR, bayes_tC, N, M, R, C, k, s, d)

        improvement = (bayes_score - greedy_score) / max(greedy_score, 1e-6) * 100

        table[key] = {
            "N": N, "M": M, "R": R, "C": C, "k": k, "s": s, "d": d,
            "greedy": {"tM": greedy_tM, "tR": greedy_tR, "tC": greedy_tC,
                       "score": round(greedy_score, 4)},
            "bayesian": {"tM": bayes_tM, "tR": bayes_tR, "tC": bayes_tC,
                         "score": round(bayes_score, 4)},
            "improvement_pct": round(improvement, 2),
        }
        print(f"    greedy  tM={greedy_tM} tR={greedy_tR} tC={greedy_tC} "
              f"score={greedy_score:.4f}")
        print(f"    bayesian tM={bayes_tM} tR={bayes_tR} tC={bayes_tC} "
              f"score={bayes_score:.4f}  ({improvement:+.1f}%)")

    return table


def main():
    """Quick demo on typical ResNet-50 layer shapes."""
    print("=== Bayesian Tiling Search Demo ===\n")
    demo_layers = [
        {"N": 64,  "M": 64,  "R": 56, "C": 56, "k": 1, "s": 1, "d": 1},
        {"N": 64,  "M": 256, "R": 56, "C": 56, "k": 1, "s": 1, "d": 1},
        {"N": 256, "M": 64,  "R": 56, "C": 56, "k": 1, "s": 1, "d": 1},
        {"N": 64,  "M": 64,  "R": 56, "C": 56, "k": 3, "s": 1, "d": 1},
        {"N": 256, "M": 512, "R": 28, "C": 28, "k": 1, "s": 1, "d": 1},
        {"N": 512, "M": 128, "R": 28, "C": 28, "k": 1, "s": 1, "d": 1},
        {"N": 128, "M": 128, "R": 28, "C": 28, "k": 3, "s": 1, "d": 1},
    ]

    table = build_tile_lookup(demo_layers)

    out_path = os.path.join(os.path.dirname(__file__), "tile_lookup.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved lookup table to {out_path}")


if __name__ == "__main__":
    main()
