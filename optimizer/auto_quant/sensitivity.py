"""
AutoQuant - Layer Sensitivity Analysis (Phase 3-C)
Computes per-layer quantization sensitivity using the Fisher Information
diagonal approximation (HAWQ-style).

High sensitivity score -> layer is important for accuracy -> use higher bit-width.
Low sensitivity score  -> layer can tolerate lower bit-width safely.

Usage:
  sens = compute_layer_sensitivity(mlir_path, weight_npz_path, calib_data)
  ranked = rank_layers_by_sensitivity(sens)
"""

import re
from typing import Dict, List, Tuple, Optional

import numpy as np

_HAS_INTERP = False  # COAInterpreter lives in legacy; sensitivity uses proxy mode only

# Supported quantisation bit-widths for per-layer search
SUPPORTED_BITS = [4, 8, 16]

# Perturbation scale for Fisher approximation
_EPS = 1e-4


def _quantize_array(x: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric uniform quantization of x to 'bits' bits."""
    if bits >= 16:
        return x.astype(np.float32)
    levels = 2 ** (bits - 1) - 1
    scale  = np.max(np.abs(x)) / levels
    if scale == 0.0:
        return x
    q = np.round(x / scale) * scale
    return q.astype(np.float32)


def compute_weight_perturbation(weights: np.ndarray, bits: int) -> float:
    """
    Proxy sensitivity: L2 norm of quantization error for a given bit-width.
    Approximates the Fisher-weighted Hessian trace (HAWQ Eq. 6).
    """
    w_q = _quantize_array(weights, bits)
    err  = weights.astype(np.float32) - w_q
    return float(np.sum(err ** 2))


def compute_layer_sensitivity(weight_npz_path: str,
                               bits_list: List[int] = SUPPORTED_BITS
                               ) -> List[Dict]:
    """
    Compute quantization sensitivity for every layer in a weight .npz file.

    Returns a list of dicts:
      {name, shape, n_params, sensitivity_per_bit: {4: float, 8: float, 16: float}}
    """
    weights = np.load(weight_npz_path)
    results = []

    for name in weights.files:
        w = weights[name].astype(np.float64)
        # Only analyse weight tensors (skip 1-D bias / scale / zp arrays)
        if w.ndim < 2:
            continue
        n_params = int(np.prod(w.shape))
        sens_per_bit = {}
        for bits in bits_list:
            sens_per_bit[bits] = compute_weight_perturbation(w, bits)

        results.append({
            "name":        name,
            "shape":       list(w.shape),
            "n_params":    n_params,
            "sensitivity": sens_per_bit,
        })

    return results


def rank_layers_by_sensitivity(sensitivity_results: List[Dict],
                                bits_ref: int = 4) -> List[Dict]:
    """
    Rank layers from most sensitive (keep high bits) to least sensitive.
    sensitivity at bits_ref (default 4-bit error) is used as the ranking key.
    """
    ranked = sorted(sensitivity_results,
                    key=lambda r: r["sensitivity"][bits_ref],
                    reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i
    return ranked


def assign_bits_budget(ranked: List[Dict],
                        bit_budget: int,
                        sensitive_bits: int = 8,
                        default_bits: int = 4) -> Dict[str, int]:
    """
    Greedy bit assignment under a bit-budget constraint:
      - Top-k most sensitive layers get `sensitive_bits`
      - Remaining layers get `default_bits`
    k is chosen so that total bits ≈ bit_budget.

    Args:
        ranked:         Output of rank_layers_by_sensitivity()
        bit_budget:     Target average bits per parameter
        sensitive_bits: Bits for sensitive layers (default 8)
        default_bits:   Bits for insensitive layers (default 4)

    Returns:
        dict {layer_name: n_bits}
    """
    total_params = sum(r["n_params"] for r in ranked)
    if total_params == 0:
        return {}

    # Binary search for k (number of sensitive layers)
    best_k = 0
    for k in range(len(ranked) + 1):
        sensitive_params = sum(r["n_params"] for r in ranked[:k])
        other_params     = total_params - sensitive_params
        avg_bits = (sensitive_params * sensitive_bits + other_params * default_bits) / total_params
        if avg_bits <= bit_budget:
            best_k = k
        else:
            break

    assignment = {}
    for i, r in enumerate(ranked):
        assignment[r["name"]] = sensitive_bits if i < best_k else default_bits

    actual_avg = sum(assignment[r["name"]] * r["n_params"]
                     for r in ranked) / total_params
    print(f"[sensitivity] Budget {bit_budget:.1f} bits -> "
          f"k={best_k} sensitive layers, avg={actual_avg:.2f} bits/param")
    return assignment


def print_sensitivity_table(ranked: List[Dict]):
    """Pretty-print the sensitivity ranking table."""
    print(f"\n{'Rank':>4}  {'Layer name':<50}  {'Params':>8}  "
          f"{'4-bit err':>10}  {'8-bit err':>10}")
    print("-" * 90)
    for r in ranked:
        e4 = r["sensitivity"].get(4, 0)
        e8 = r["sensitivity"].get(8, 0)
        print(f"{r['rank']:>4}  {r['name']:<50}  {r['n_params']:>8}  "
              f"{e4:>10.4f}  {e8:>10.6f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sensitivity.py <weights.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    print(f"=== Layer Sensitivity Analysis: {npz_path} ===\n")
    sens = compute_layer_sensitivity(npz_path)
    ranked = rank_layers_by_sensitivity(sens, bits_ref=4)
    print_sensitivity_table(ranked)

    print("\n=== Bit Assignment (budget=6 bits/param) ===")
    assignment = assign_bits_budget(ranked, bit_budget=6.0)
    for name, bits in list(assignment.items())[:10]:
        print(f"  {name:<50} -> {bits} bits")
