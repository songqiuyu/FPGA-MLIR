"""
coa.pruning — Structural pruning interface (stub + L1-norm filter pruning).

Provides a framework for structured pruning of ONNX models **before**
quantization.  The pruned float ONNX can then be fed into ``coa.quantize``
for PTQ and finally into ``coa.onnx_importer`` for COA MLIR generation.

Currently implemented
---------------------
* **L1-norm filter pruning** — remove output channels with smallest L1 norm.

Planned / extensible
--------------------
* Taylor first-order importance (requires gradient proxy or calibration).
* FPGM (Filter Pruning via Geometric Median).
* Slim-style BN γ pruning.
* Hardware-aware latency-driven pruning (integrated with ``coa.tiling``).

Usage
~~~~~
>>> from coa.pruning import prune_onnx, PruningConfig, PruningMethod
>>> cfg = PruningConfig(method=PruningMethod.L1_NORM, ratio=0.25)
>>> prune_onnx("resnet18.onnx", "resnet18_pruned.onnx", config=cfg)

Dependencies: ``onnx >= 1.14``, ``numpy >= 1.24``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import onnx
    from onnx import ModelProto, TensorProto, helper, numpy_helper
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


# ────────────────────────────────────────────────────────────────
# Public config
# ────────────────────────────────────────────────────────────────

class PruningMethod(str, Enum):
    L1_NORM = "l1_norm"
    # Future:
    # TAYLOR  = "taylor"
    # FPGM    = "fpgm"
    # BN_SLIM = "bn_slim"


@dataclass
class PruningConfig:
    """Configuration for structural pruning."""
    method: PruningMethod = PruningMethod.L1_NORM
    ratio: float = 0.25
    skip_layers: Set[str] = field(default_factory=set)
    min_channels: int = 16
    round_to: int = 16


# ────────────────────────────────────────────────────────────────
# Importance scoring
# ────────────────────────────────────────────────────────────────

def l1_norm_importance(W: np.ndarray) -> np.ndarray:
    """
    Per-output-channel L1 norm importance score.
    W shape: (C_out, C_in, kH, kW) for Conv, (C_out, C_in) for Gemm.
    Returns: (C_out,) importance scores.
    """
    axes = tuple(range(1, W.ndim))
    return np.sum(np.abs(W), axis=axes)


_IMPORTANCE_FN = {
    PruningMethod.L1_NORM: l1_norm_importance,
}


# ────────────────────────────────────────────────────────────────
# Channel selection
# ────────────────────────────────────────────────────────────────

def select_channels(
    importance: np.ndarray,
    ratio: float,
    min_channels: int = 16,
    round_to: int = 16,
) -> np.ndarray:
    """
    Return sorted indices of channels to **keep** after pruning.

    Parameters
    ----------
    importance : (C,) per-channel importance scores.
    ratio : fraction of channels to **remove** (0 to 1).
    min_channels : minimum number of channels to keep.
    round_to : round kept count to multiple of this (FPGA-friendly).

    Returns
    -------
    keep_indices : sorted 1-D int array of channels to keep.
    """
    C = len(importance)
    n_remove = int(C * ratio)
    n_keep = max(C - n_remove, min_channels)
    # Round up to nearest multiple of round_to
    if round_to > 1:
        n_keep = max(min_channels, ((n_keep + round_to - 1) // round_to) * round_to)
    n_keep = min(n_keep, C)
    ranked = np.argsort(importance)[::-1]  # descending
    keep = np.sort(ranked[:n_keep])
    return keep


# ────────────────────────────────────────────────────────────────
# ONNX graph surgery
# ────────────────────────────────────────────────────────────────

def _find_conv_nodes(graph) -> List:
    return [n for n in graph.node if n.op_type == "Conv"]


def _get_init(graph, name: str) -> Optional[np.ndarray]:
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _set_init(graph, name: str, arr: np.ndarray):
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer.remove(init)
            break
    graph.initializer.append(numpy_helper.from_array(arr, name=name))


def _prune_conv_output_channels(
    graph, node, keep: np.ndarray,
) -> None:
    """Prune output channels of a Conv node (weight axis=0, bias axis=0)."""
    w_name = node.input[1]
    W = _get_init(graph, w_name)
    if W is None:
        return
    W_pruned = W[keep]
    _set_init(graph, w_name, W_pruned)

    # Prune bias if present
    if len(node.input) > 2 and node.input[2]:
        b_name = node.input[2]
        B = _get_init(graph, b_name)
        if B is not None:
            _set_init(graph, b_name, B[keep])


def _prune_conv_input_channels(
    graph, node, keep: np.ndarray,
) -> None:
    """Prune input channels of a Conv node (weight axis=1)."""
    w_name = node.input[1]
    W = _get_init(graph, w_name)
    if W is None:
        return
    W_pruned = W[:, keep]
    _set_init(graph, w_name, W_pruned)


def _prune_bn_channels(graph, node, keep: np.ndarray) -> None:
    """Prune BatchNormalization parameters (all 1-D, axis=0)."""
    for idx in range(1, min(len(node.input), 5)):
        name = node.input[idx]
        arr = _get_init(graph, name)
        if arr is not None and arr.ndim == 1:
            _set_init(graph, name, arr[keep])


def _build_output_consumer_map(graph) -> Dict[str, List]:
    """Map: tensor_name → list of (node, input_index) that consume it."""
    consumers: Dict[str, List] = {}
    for node in graph.node:
        for idx, inp in enumerate(node.input):
            if inp:
                consumers.setdefault(inp, []).append((node, idx))
    return consumers


# ────────────────────────────────────────────────────────────────
# Main pruning logic
# ────────────────────────────────────────────────────────────────

def prune_model(
    model: "ModelProto",
    config: PruningConfig,
    verbose: bool = True,
) -> "ModelProto":
    """
    Apply structured pruning to an ONNX model (in-place on a deep copy).

    Currently supports pruning output filters of Conv nodes and propagating
    the channel change to downstream Conv input channels and BN parameters.

    Returns the pruned ModelProto.
    """
    model = copy.deepcopy(model)
    graph = model.graph
    importance_fn = _IMPORTANCE_FN[config.method]

    conv_nodes = _find_conv_nodes(graph)
    consumers = _build_output_consumer_map(graph)

    pruned_count = 0
    total_removed = 0

    for conv in conv_nodes:
        w_name = conv.input[1]
        if w_name in config.skip_layers or conv.name in config.skip_layers:
            continue

        W = _get_init(graph, w_name)
        if W is None or W.ndim < 2:
            continue

        C_out = W.shape[0]
        scores = importance_fn(W)
        keep = select_channels(scores, config.ratio,
                               config.min_channels, config.round_to)

        if len(keep) >= C_out:
            continue

        n_removed = C_out - len(keep)
        total_removed += n_removed
        pruned_count += 1

        if verbose:
            print(f"[pruning] {conv.name or w_name}: {C_out} → {len(keep)} channels "
                  f"(-{n_removed})")

        # 1. Prune this Conv's output channels
        _prune_conv_output_channels(graph, conv, keep)

        # 2. Propagate to consumers of this Conv's output
        out_tensor = conv.output[0]
        for consumer_node, _ in consumers.get(out_tensor, []):
            if consumer_node.op_type == "Conv":
                _prune_conv_input_channels(graph, consumer_node, keep)
            elif consumer_node.op_type == "BatchNormalization":
                _prune_bn_channels(graph, consumer_node, keep)
            # Add / Concat / etc. — more complex propagation needed
            # For now, skip (user should verify model correctness)

    if verbose:
        print(f"[pruning] Pruned {pruned_count} layers, removed {total_removed} filters total")

    return model


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def prune_onnx(
    input_onnx: str,
    output_onnx: str,
    config: Optional[PruningConfig] = None,
    verbose: bool = True,
) -> str:
    """
    Load a float ONNX, apply structural pruning, save the result.

    Parameters
    ----------
    input_onnx : path to input float32 ONNX model.
    output_onnx : path to write pruned ONNX.
    config : PruningConfig (default: L1 norm, 25 % ratio).
    verbose : print progress.

    Returns
    -------
    output_onnx path.
    """
    if not _HAS_ONNX:
        raise ImportError("onnx package required: pip install onnx")

    if config is None:
        config = PruningConfig()

    if verbose:
        print(f"[coa.pruning] Loading {input_onnx}")
        print(f"[coa.pruning] Method: {config.method.value}, ratio: {config.ratio}")

    model = onnx.load(input_onnx)
    model = prune_model(model, config, verbose=verbose)
    onnx.save(model, output_onnx)

    if verbose:
        print(f"[coa.pruning] Saved pruned model to {output_onnx}")

    return output_onnx


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="COA Structural Pruning — prune float ONNX before quantization")
    parser.add_argument("input", help="Input float32 ONNX model")
    parser.add_argument("output", help="Output pruned ONNX model")
    parser.add_argument("--method", choices=["l1_norm"], default="l1_norm")
    parser.add_argument("--ratio", type=float, default=0.25,
                        help="Fraction of channels to remove (default: 0.25)")
    parser.add_argument("--min-channels", type=int, default=16)
    parser.add_argument("--round-to", type=int, default=16,
                        help="Round kept channels to multiple of N (default: 16)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Weight names or node names to skip")
    args = parser.parse_args()

    cfg = PruningConfig(
        method=PruningMethod(args.method),
        ratio=args.ratio,
        min_channels=args.min_channels,
        round_to=args.round_to,
        skip_layers=set(args.skip),
    )
    prune_onnx(args.input, args.output, config=cfg)


if __name__ == "__main__":
    main()
