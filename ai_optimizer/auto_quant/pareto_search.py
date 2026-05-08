"""
AutoQuant - NSGA-II Multi-Objective Pareto Search (Phase 3-C)

Searches the mixed-precision quantization space using NSGA-II with three objectives:
  1. Minimize accuracy loss    (proxy: sum of Fisher sensitivity × bits mismatch)
  2. Minimize VLIW instruction overhead (proxy: proportional to avg bit-width)
  3. Minimize peak buffer utilization   (proxy: proportional to max layer weight size)

After search, plots (or saves) the Pareto frontier and writes the best
assignments to mixed_precision_config.json.

Requirements: pymoo (pip install pymoo), numpy
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sensitivity import (compute_layer_sensitivity, rank_layers_by_sensitivity,
                          SUPPORTED_BITS)

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.termination import get_termination
    _HAS_PYMOO = True
except ImportError:
    _HAS_PYMOO = False
    print("[pareto_search] pymoo not found. Install: pip install pymoo")

# Map integer gene [0, 1, 2] -> bit-width
BITS_MAP = {0: 4, 1: 8, 2: 16}


class MixedPrecisionProblem(ElementwiseProblem):
    """
    Multi-objective optimisation problem for mixed-precision quantization.

    Decision variable x[i] ∈ {0,1,2} -> bit-width index for layer i.

    Objectives (all to minimise):
      f1: Normalised accuracy proxy loss (Fisher sensitivity weighted error)
      f2: Normalised avg bit-width (hardware overhead proxy)
      f3: Normalised peak weight buffer size (memory pressure)
    """

    def __init__(self, sensitivity_results: List[Dict]):
        self.layers   = sensitivity_results
        n_layers      = len(sensitivity_results)
        n_bits_choices = len(BITS_MAP)

        super().__init__(
            n_var=n_layers,
            n_obj=3,
            n_constr=0,
            xl=np.zeros(n_layers, dtype=int),
            xu=np.full(n_layers, n_bits_choices - 1, dtype=int),
            vtype=int,
        )

        # Precompute sensitivity norms for objective scaling
        max_sens = max(
            max(r["sensitivity"].values()) for r in self.layers
        ) or 1.0
        self._max_sens = max_sens
        self._total_params = sum(r["n_params"] for r in self.layers) or 1

    def _evaluate(self, x, out, *args, **kwargs):
        # ---- Objective 1: accuracy loss proxy ----
        acc_loss = 0.0
        for i, gene in enumerate(x):
            bits = BITS_MAP[int(gene)]
            # Higher bits = lower quantization error; use sensitivity * q_error
            sens_4bit = self.layers[i]["sensitivity"].get(4, 0)
            sens_bits  = self.layers[i]["sensitivity"].get(bits, 0)
            # Accuracy penalty: extra error ABOVE 16-bit baseline, weighted by sensitivity
            acc_loss += sens_bits / (self._max_sens + 1e-8)
        acc_loss /= len(self.layers)

        # ---- Objective 2: hardware overhead (avg bits / max_bits) ----
        total_bits = sum(
            BITS_MAP[int(gene)] * self.layers[i]["n_params"]
            for i, gene in enumerate(x)
        )
        avg_bits     = total_bits / self._total_params
        hw_overhead  = avg_bits / 16.0  # normalised to [0, 1]

        # ---- Objective 3: peak buffer pressure ----
        # Approximate: max layer weight size in bytes
        max_bytes = max(
            self.layers[i]["n_params"] * BITS_MAP[int(x[i])] / 8
            for i in range(len(x))
        ) if self.layers else 1.0
        # Normalise by 32MB
        mem_pressure = max_bytes / (32 * 1024 * 1024)

        out["F"] = [acc_loss, hw_overhead, mem_pressure]


def run_pareto_search(weight_npz_path: str,
                      n_gen: int = 100,
                      pop_size: int = 50,
                      save_dir: str = ".") -> List[Dict]:
    """
    Run NSGA-II Pareto search and return the Pareto-optimal configurations.

    Returns list of dicts:
      {bit_assignment: {layer: bits}, objectives: [acc_loss, hw, mem]}
    """
    print(f"=== AutoQuant Pareto Search ===")
    print(f"Weight file : {weight_npz_path}")
    print(f"NSGA-II: {n_gen} generations × {pop_size} individuals\n")

    # Compute sensitivity
    sens = compute_layer_sensitivity(weight_npz_path)
    if not sens:
        print("[pareto_search] No weight layers found in npz file.")
        return []
    ranked = rank_layers_by_sensitivity(sens)

    if not _HAS_PYMOO:
        print("[pareto_search] pymoo not available; running greedy fallback.")
        return _greedy_pareto_fallback(ranked, save_dir)

    problem = MixedPrecisionProblem(ranked)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=None),
        mutation=PM(eta=20, vtype=float, repair=None),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)
    result = pymoo_minimize(problem, algorithm, termination,
                            seed=42, verbose=True)

    # Parse Pareto front
    pareto_configs = []
    for i, x in enumerate(result.X):
        assignment = {
            ranked[j]["name"]: BITS_MAP[int(x[j])]
            for j in range(len(ranked))
        }
        f = result.F[i]
        pareto_configs.append({
            "id":             i,
            "bit_assignment": assignment,
            "objectives": {
                "acc_loss":     float(f[0]),
                "hw_overhead":  float(f[1]),
                "mem_pressure": float(f[2]),
            },
            "avg_bits": float(sum(assignment.values()) / max(len(assignment), 1)),
        })

    # Sort by accuracy loss (ascending) for display
    pareto_configs.sort(key=lambda c: c["objectives"]["acc_loss"])

    # Save Pareto front
    out_path = os.path.join(save_dir, "pareto_front.json")
    with open(out_path, "w") as f:
        json.dump(pareto_configs, f, indent=2)
    print(f"\nPareto front ({len(pareto_configs)} solutions) saved to {out_path}")

    # Print summary table
    print(f"\n{'ID':>4}  {'avg_bits':>8}  {'acc_loss':>10}  {'hw':>8}  {'mem':>8}")
    print("-" * 50)
    for c in pareto_configs[:20]:  # show top 20
        print(f"{c['id']:>4}  {c['avg_bits']:>8.2f}  "
              f"{c['objectives']['acc_loss']:>10.4f}  "
              f"{c['objectives']['hw_overhead']:>8.4f}  "
              f"{c['objectives']['mem_pressure']:>8.4f}")

    # Save the best accuracy-preserving config separately
    best = pareto_configs[0]
    best_path = os.path.join(save_dir, "mixed_precision_config.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest accuracy config saved to {best_path}")
    print(f"  avg bits = {best['avg_bits']:.2f}, acc_loss = {best['objectives']['acc_loss']:.4f}")

    return pareto_configs


def _greedy_pareto_fallback(ranked: List[Dict], save_dir: str) -> List[Dict]:
    """Simple greedy Pareto approximation when pymoo is unavailable."""
    configs = []
    for target_bits in [4, 6, 8, 10, 12, 16]:
        from sensitivity import assign_bits_budget
        assignment = assign_bits_budget(ranked, bit_budget=float(target_bits))
        avg = sum(assignment.values()) / max(len(assignment), 1)
        # Proxy objectives
        acc = sum(
            ranked[i]["sensitivity"].get(assignment.get(r["name"], 8), 0)
            for i, r in enumerate(ranked)
        ) / (len(ranked) or 1)
        configs.append({
            "id": target_bits,
            "bit_assignment": assignment,
            "objectives": {
                "acc_loss":     float(acc),
                "hw_overhead":  avg / 16.0,
                "mem_pressure": avg / 16.0,
            },
            "avg_bits": float(avg),
        })
        print(f"  target_bits={target_bits}  avg={avg:.2f}  acc_loss={acc:.4f}")

    out_path = os.path.join(save_dir, "pareto_front.json")
    with open(out_path, "w") as f:
        json.dump(configs, f, indent=2)
    return configs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pareto_search.py <weights.npz> [n_gen] [pop_size]")
        sys.exit(1)
    npz   = sys.argv[1]
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    pop   = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    save  = os.path.dirname(npz) or "."
    run_pareto_search(npz, n_gen=n_gen, pop_size=pop, save_dir=save)
