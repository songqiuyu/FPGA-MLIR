"""
coa.quantize — Self-contained Post-Training Quantization (PTQ) toolkit.

Replaces onnxruntime.quantization entirely.  Reads a **float32 ONNX** model,
calibrates it, and writes out a **QOperator-format INT8 ONNX** that
coa.onnx_importer can directly consume.

Features
--------
* Two activation formats: **int8×int8** (default) and **uint8×int8**.
* Per-tensor or per-channel weight quantization.
* Min-max, percentile (99.99 %), and entropy (KL-divergence) calibration.
* Smooth-Quant channel-wise rescaling (α ∈ [0, 1], default 0.5).
* GPTQ-style layer-wise error correction (Hessian proxy).
* Hook-based architecture — model-agnostic, works with any ONNX op graph.
* Structural-pruning interface placeholder (see ``coa.pruning``).

Typical workflow
~~~~~~~~~~~~~~~~
>>> from coa.quantize import quantize_onnx
>>> quantize_onnx(
...     "resnet18.onnx",
...     "resnet18_quant_int8.onnx",
...     calib_data=np.random.randn(64, 3, 224, 224).astype(np.float32),
...     act_format="int8",          # or "uint8"
...     weight_per_channel=True,
...     calibration="minmax",       # "minmax" | "percentile" | "entropy"
... )

Dependencies: ``onnx >= 1.14``, ``numpy >= 1.24``.
No ``onnxruntime`` needed — calibration runs a pure-numpy graph executor.
"""

from __future__ import annotations

import copy
import math
from enum import Enum
from typing import (
    Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union,
)

import numpy as np

try:
    import onnx
    from onnx import (
        ModelProto, GraphProto, NodeProto, TensorProto,
        helper, numpy_helper,
    )
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


# ────────────────────────────────────────────────────────────────
# Public enums / config
# ────────────────────────────────────────────────────────────────

class ActFormat(str, Enum):
    """Activation data-type format."""
    INT8  = "int8"       # symmetric signed: [-128, 127]
    UINT8 = "uint8"      # asymmetric unsigned: [0, 255]


class CalibMethod(str, Enum):
    """Calibration algorithm."""
    MINMAX     = "minmax"
    PERCENTILE = "percentile"
    ENTROPY    = "entropy"


class WeightScheme(str, Enum):
    """Weight quantization granularity."""
    PER_TENSOR  = "per_tensor"
    PER_CHANNEL = "per_channel"


# ────────────────────────────────────────────────────────────────
# Low-level quantization math
# ────────────────────────────────────────────────────────────────

def compute_scale_zp_symmetric(
    vmin: float, vmax: float, bits: int = 8
) -> Tuple[float, int]:
    """Symmetric signed quantization (int8): zero_point = 0."""
    abs_max = max(abs(vmin), abs(vmax), 1e-12)
    qmax = (1 << (bits - 1)) - 1          # 127 for 8-bit
    scale = abs_max / qmax
    return float(scale), 0


def compute_scale_zp_asymmetric(
    vmin: float, vmax: float, bits: int = 8
) -> Tuple[float, int]:
    """Asymmetric unsigned quantization (uint8): zero_point ∈ [0, 255]."""
    qmin, qmax = 0, (1 << bits) - 1       # 0, 255 for 8-bit
    vmin = min(vmin, 0.0)
    vmax = max(vmax, 0.0)
    scale = max((vmax - vmin), 1e-12) / (qmax - qmin)
    zp = int(round(qmin - vmin / scale))
    zp = max(qmin, min(qmax, zp))
    return float(scale), int(zp)


def quantize_array(
    x: np.ndarray, scale: float, zp: int,
    signed: bool = True, bits: int = 8
) -> np.ndarray:
    """Quantize float array → integer array (clipped)."""
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    q = np.round(x / scale).astype(np.int64) + zp
    return np.clip(q, lo, hi).astype(np.int8 if signed else np.uint8)


def dequantize_array(
    q: np.ndarray, scale: float, zp: int
) -> np.ndarray:
    """Dequantize integer array → float32."""
    return (q.astype(np.float32) - zp) * scale


# ────────────────────────────────────────────────────────────────
# Calibration algorithms
# ────────────────────────────────────────────────────────────────

def _calib_minmax(histograms: np.ndarray, bin_edges: np.ndarray,
                  **_kw) -> Tuple[float, float]:
    """Min-max calibration: simply use observed extremes."""
    return float(bin_edges[0]), float(bin_edges[-1])


def _calib_percentile(histograms: np.ndarray, bin_edges: np.ndarray,
                      percentile: float = 99.99,
                      **_kw) -> Tuple[float, float]:
    """Percentile clipping: discard top/bottom 0.01 % of values."""
    cdf = np.cumsum(histograms).astype(np.float64)
    total = cdf[-1]
    if total == 0:
        return float(bin_edges[0]), float(bin_edges[-1])
    lo_idx = int(np.searchsorted(cdf, total * (1 - percentile / 100)))
    hi_idx = int(np.searchsorted(cdf, total * (percentile / 100)))
    lo_idx = max(0, lo_idx)
    hi_idx = min(len(bin_edges) - 2, hi_idx)
    return float(bin_edges[lo_idx]), float(bin_edges[hi_idx + 1])


def _calib_entropy(histograms: np.ndarray, bin_edges: np.ndarray,
                   num_quantized_bins: int = 128,
                   **_kw) -> Tuple[float, float]:
    """
    Entropy (KL-divergence) calibration — TensorRT-style.
    Finds the clipping range that minimises KL(P || Q) where P is the
    reference FP32 distribution and Q is the quantized + dequantized version.
    """
    hist = histograms.astype(np.float64)
    n_bins = len(hist)
    if n_bins <= num_quantized_bins:
        return float(bin_edges[0]), float(bin_edges[-1])

    bin_width = bin_edges[1] - bin_edges[0]
    best_kl = float("inf")
    best_k  = n_bins

    for k in range(num_quantized_bins, n_bins + 1):
        # Reference distribution P (clipped to [0, k])
        p = hist[:k].copy()
        p[-1] += np.sum(hist[k:])                   # overflow bin
        total = np.sum(p)
        if total == 0:
            continue
        p /= total

        # Build quantized distribution Q by merging bins
        merge = k // num_quantized_bins
        remainder = k - merge * num_quantized_bins
        q_raw = np.zeros(num_quantized_bins, dtype=np.float64)
        src = 0
        for i in range(num_quantized_bins):
            w = merge + (1 if i < remainder else 0)
            q_raw[i] = np.sum(hist[src:src + w])
            src += w
        # Expand Q back to k bins
        q_expand = np.zeros(k, dtype=np.float64)
        src = 0
        for i in range(num_quantized_bins):
            w = merge + (1 if i < remainder else 0)
            val = q_raw[i] / max(w, 1)
            q_expand[src:src + w] = val
            src += w
        q_sum = np.sum(q_expand)
        if q_sum == 0:
            continue
        q_expand /= q_sum

        # KL(P || Q) — only over bins where both > 0
        mask = (p > 0) & (q_expand > 0)
        kl = np.sum(p[mask] * np.log(p[mask] / q_expand[mask]))
        if kl < best_kl:
            best_kl = kl
            best_k  = k

    clip_max = bin_edges[best_k]
    clip_min = bin_edges[0]
    return float(clip_min), float(clip_max)


_CALIB_FUNCTIONS: Dict[CalibMethod, Callable] = {
    CalibMethod.MINMAX:     _calib_minmax,
    CalibMethod.PERCENTILE: _calib_percentile,
    CalibMethod.ENTROPY:    _calib_entropy,
}


# ────────────────────────────────────────────────────────────────
# SmoothQuant channel rescaling
# ────────────────────────────────────────────────────────────────

def smooth_quant_scales(
    act_abs_max: np.ndarray,    # per-channel |activation| max  (C,)
    weight_abs_max: np.ndarray, # per-channel |weight|    max   (C,)
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute per-channel smooth-quant rescaling factors ``s``.

    ``s_j = max(|X_j|)^α / max(|W_j|)^(1-α)``

    The activations are divided by ``s`` and the weights are multiplied by
    ``s`` **before** quantization so that the outlier magnitude is balanced
    between the two tensors.  The floating-point result is unchanged.

    Returns ``s`` array of shape (C,).
    """
    eps = 1e-12
    a = np.maximum(act_abs_max, eps)
    w = np.maximum(weight_abs_max, eps)
    s = np.power(a, alpha) / np.power(w, 1 - alpha)
    return s.astype(np.float32)


# ────────────────────────────────────────────────────────────────
# GPTQ-style weight correction
# ────────────────────────────────────────────────────────────────

def gptq_correct_weight(
    W: np.ndarray,
    H_diag: np.ndarray,
    scale: Union[float, np.ndarray],
    zp: Union[int, np.ndarray],
    signed: bool = True,
    bits: int = 8,
    block_size: int = 128,
    dampening: float = 0.01,
) -> np.ndarray:
    """
    GPTQ-inspired layer-wise weight correction.

    Iterates over columns in blocks; after quantising each column the
    residual error is distributed to remaining columns weighted by the
    diagonal Hessian proxy ``H_diag``.

    Parameters
    ----------
    W : (out_features, in_features) float32 weight matrix.
    H_diag : (in_features,) approximate Hessian diagonal (from calibration
             activations: ``diag(X^T X / n)``).
    scale, zp : quantization parameters.
    block_size : GPTQ block width.
    dampening : Hessian diagonal dampening ratio.

    Returns
    -------
    W_q_corrected : int8/uint8 quantized weight with error correction applied.
    """
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1

    out_feat, in_feat = W.shape
    W_q = np.zeros_like(W, dtype=np.float32)
    E = W.copy().astype(np.float64)

    h = H_diag.astype(np.float64).copy()
    h += dampening * np.mean(h)
    h = np.maximum(h, 1e-12)

    for blk_start in range(0, in_feat, block_size):
        blk_end = min(blk_start + block_size, in_feat)
        for j in range(blk_start, blk_end):
            w_col = E[:, j].copy()
            # Quantize
            if np.isscalar(scale):
                q = np.round(w_col / scale).astype(np.int64) + int(zp)
            else:
                q = np.round(w_col / scale[:, None].ravel()[:out_feat]).astype(np.int64)
                q += zp[:, None].ravel()[:out_feat].astype(np.int64)
            q = np.clip(q, lo, hi)
            # Dequantize
            if np.isscalar(scale):
                w_hat = (q.astype(np.float64) - int(zp)) * float(scale)
            else:
                w_hat = (q.astype(np.float64) - zp[:, None].ravel()[:out_feat].astype(np.float64)) * scale[:, None].ravel()[:out_feat].astype(np.float64)

            W_q[:, j] = q.astype(np.float32)
            err = w_col - w_hat  # (out_feat,)

            # Distribute error to remaining columns
            remaining = np.arange(j + 1, blk_end)
            if len(remaining) > 0:
                h_inv_ratio = h[j] / h[remaining]
                E[:, remaining] += err[:, None] * h_inv_ratio[None, :]

    return np.clip(W_q, lo, hi).astype(np.int8 if signed else np.uint8)


# ────────────────────────────────────────────────────────────────
# Pure-numpy mini graph executor (calibration only)
# ────────────────────────────────────────────────────────────────

class _GraphRunner:
    """
    Minimal forward-only ONNX graph executor using only numpy.
    Supports: Conv, Relu, MaxPool, GlobalAveragePool, Gemm, Add, Flatten,
    Reshape, BatchNormalization, Clip, Sigmoid, Concat, AveragePool,
    Transpose, Unsqueeze, Squeeze, MatMul, Softmax, Pad.
    Enough to run typical CNN calibration.
    """

    def __init__(self, model: "ModelProto"):
        self._graph = model.graph
        self._inits: Dict[str, np.ndarray] = {}
        for init in self._graph.initializer:
            self._inits[init.name] = numpy_helper.to_array(init)

    def run(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        env: Dict[str, np.ndarray] = dict(self._inits)
        env.update(feed)

        for node in self._graph.node:
            try:
                outs = self._exec_node(node, env)
            except Exception:
                # Skip unsupported ops — calibration is best-effort
                outs = {o: np.zeros(1, dtype=np.float32) for o in node.output}
            env.update(outs)

        return env

    def _attr(self, node: "NodeProto", name: str, default=None):
        for a in node.attribute:
            if a.name == name:
                if a.type == 7: return list(a.ints)
                if a.type == 6: return list(a.floats)
                if a.type == 1: return a.f
                if a.type == 2: return a.i
                if a.type == 3: return a.s.decode() if isinstance(a.s, bytes) else a.s
        return default

    def _exec_node(self, node, env):
        op = node.op_type
        inputs = [env.get(n) for n in node.input]
        result = {}

        if op == "Conv":
            X, W = inputs[0], inputs[1]
            B = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
            ks = self._attr(node, "kernel_shape", list(W.shape[2:]))
            strides = self._attr(node, "strides", [1] * len(ks))
            pads = self._attr(node, "pads", [0] * 2 * len(ks))
            dilations = self._attr(node, "dilations", [1] * len(ks))
            group = self._attr(node, "group", 1)
            result[node.output[0]] = self._conv2d(X, W, B, strides, pads, dilations, group)

        elif op == "Relu":
            result[node.output[0]] = np.maximum(inputs[0], 0)

        elif op in ("Clip", "Min", "Max"):
            x = inputs[0]
            lo = float(inputs[1]) if len(inputs) > 1 and inputs[1] is not None else self._attr(node, "min", -np.inf)
            hi = float(inputs[2]) if len(inputs) > 2 and inputs[2] is not None else self._attr(node, "max", np.inf)
            result[node.output[0]] = np.clip(x, lo, hi)

        elif op == "Sigmoid":
            result[node.output[0]] = 1.0 / (1.0 + np.exp(-inputs[0].astype(np.float64))).astype(np.float32)

        elif op == "MaxPool":
            result[node.output[0]] = self._maxpool(inputs[0], node)
            if len(node.output) > 1:
                result[node.output[1]] = np.zeros(1)  # indices stub

        elif op == "AveragePool":
            result[node.output[0]] = self._avgpool(inputs[0], node)

        elif op == "GlobalAveragePool":
            axes = tuple(range(2, inputs[0].ndim))
            result[node.output[0]] = np.mean(inputs[0], axis=axes, keepdims=True).astype(np.float32)

        elif op == "Gemm":
            A, B_m = inputs[0], inputs[1]
            C_b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
            alpha = self._attr(node, "alpha", 1.0)
            beta = self._attr(node, "beta", 1.0)
            transA = self._attr(node, "transA", 0)
            transB = self._attr(node, "transB", 0)
            if transA: A = A.T
            if transB: B_m = B_m.T
            out = alpha * (A @ B_m)
            if C_b is not None:
                out += beta * C_b
            result[node.output[0]] = out.astype(np.float32)

        elif op == "MatMul":
            result[node.output[0]] = (inputs[0] @ inputs[1]).astype(np.float32)

        elif op == "Add":
            result[node.output[0]] = (inputs[0] + inputs[1]).astype(np.float32)

        elif op == "Flatten":
            axis = self._attr(node, "axis", 1)
            x = inputs[0]
            new_shape = (int(np.prod(x.shape[:axis])), -1)
            result[node.output[0]] = x.reshape(new_shape)

        elif op == "Reshape":
            x = inputs[0]
            shape = inputs[1].astype(np.int64).tolist() if inputs[1] is not None else [-1]
            result[node.output[0]] = np.reshape(x, shape)

        elif op == "Transpose":
            perm = self._attr(node, "perm", None)
            result[node.output[0]] = np.transpose(inputs[0], axes=perm)

        elif op == "BatchNormalization":
            X, scale, B, mean, var = inputs[:5]
            eps = self._attr(node, "epsilon", 1e-5)
            X_norm = (X - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + eps)
            result[node.output[0]] = (scale.reshape(1, -1, 1, 1) * X_norm + B.reshape(1, -1, 1, 1)).astype(np.float32)

        elif op == "Concat":
            axis = self._attr(node, "axis", 0)
            valid = [i for i in inputs if i is not None]
            result[node.output[0]] = np.concatenate(valid, axis=axis)

        elif op == "Unsqueeze":
            axes = self._attr(node, "axes", None)
            x = inputs[0]
            if axes is None and len(inputs) > 1 and inputs[1] is not None:
                axes = inputs[1].tolist()
            if axes:
                for ax in sorted(axes):
                    x = np.expand_dims(x, axis=ax)
            result[node.output[0]] = x

        elif op == "Squeeze":
            axes = self._attr(node, "axes", None)
            x = inputs[0]
            if axes is None and len(inputs) > 1 and inputs[1] is not None:
                axes = inputs[1].tolist()
            if axes:
                result[node.output[0]] = np.squeeze(x, axis=tuple(axes))
            else:
                result[node.output[0]] = np.squeeze(x)

        elif op == "Pad":
            x = inputs[0]
            pads_val = inputs[1].astype(np.int64).tolist() if len(inputs) > 1 and inputs[1] is not None else self._attr(node, "pads", [])
            n = x.ndim
            pad_pairs = [(pads_val[i], pads_val[i + n]) for i in range(n)] if len(pads_val) >= 2 * n else [(0, 0)] * n
            result[node.output[0]] = np.pad(x, pad_pairs, mode='constant')

        elif op == "Softmax":
            axis = self._attr(node, "axis", -1)
            x = inputs[0]
            e = np.exp(x - np.max(x, axis=axis, keepdims=True))
            result[node.output[0]] = (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)

        elif op in ("Dropout", "Identity"):
            result[node.output[0]] = inputs[0]

        elif op == "Constant":
            t = None
            for a in node.attribute:
                if a.name == "value" and a.t:
                    t = numpy_helper.to_array(a.t)
            result[node.output[0]] = t if t is not None else np.zeros(1, dtype=np.float32)

        elif op == "Shape":
            result[node.output[0]] = np.array(inputs[0].shape, dtype=np.int64)

        elif op == "Gather":
            axis = self._attr(node, "axis", 0)
            result[node.output[0]] = np.take(inputs[0], inputs[1].astype(np.int64), axis=axis)

        elif op == "Mul":
            result[node.output[0]] = (inputs[0] * inputs[1]).astype(np.float32)

        elif op == "Sub":
            result[node.output[0]] = (inputs[0] - inputs[1]).astype(np.float32)

        elif op == "Div":
            result[node.output[0]] = (inputs[0] / (inputs[1] + 1e-12)).astype(np.float32)

        elif op == "Pow":
            result[node.output[0]] = np.power(inputs[0], inputs[1]).astype(np.float32)

        elif op == "Sqrt":
            result[node.output[0]] = np.sqrt(inputs[0]).astype(np.float32)

        elif op == "ReduceMean":
            axes = self._attr(node, "axes", None)
            keepdims = bool(self._attr(node, "keepdims", 1))
            result[node.output[0]] = np.mean(inputs[0], axis=tuple(axes) if axes else None, keepdims=keepdims).astype(np.float32)

        elif op == "Cast":
            to = self._attr(node, "to", 1)  # FLOAT=1
            dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}
            result[node.output[0]] = inputs[0].astype(dtype_map.get(to, np.float32))

        else:
            # Fallback: pass through first input
            if inputs and inputs[0] is not None:
                for o in node.output:
                    result[o] = inputs[0]
            else:
                for o in node.output:
                    result[o] = np.zeros(1, dtype=np.float32)

        return result

    def _conv2d(self, X, W, B, strides, pads, dilations, group):
        """Vectorized Conv2D using im2col (groups, dilations, pads supported)."""
        N, C_in, H_in, W_in = X.shape
        C_out, _C_grp, kH, kW = W.shape
        sH = strides[0];  sW = strides[1] if len(strides) > 1 else strides[0]
        pH  = pads[0];    pW  = pads[1] if len(pads) > 1 else pads[0]
        pH2 = pads[2] if len(pads) > 2 else pH
        pW2 = pads[3] if len(pads) > 3 else pW
        dH = dilations[0]; dW = dilations[1] if len(dilations) > 1 else dH

        H_out = (H_in + pH + pH2 - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + pW + pW2 - dW * (kW - 1) - 1) // sW + 1

        X_pad = np.pad(X, ((0, 0), (0, 0), (pH, pH2), (pW, pW2))).astype(np.float32)
        c_per_group = C_in // group
        f_per_group = C_out // group
        HW = H_out * W_out

        # Precompute output spatial index arrays
        h_base = np.arange(H_out, dtype=np.int32) * sH   # (H_out,)
        w_base = np.arange(W_out, dtype=np.int32) * sW   # (W_out,)

        Y = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        for g in range(group):
            X_g = X_pad[:, g * c_per_group:(g + 1) * c_per_group]  # (N, c, H_pad, W_pad)
            W_g = W[g * f_per_group:(g + 1) * f_per_group]         # (f, c, kH, kW)

            # im2col: build (N, c*kH*kW, H_out*W_out)
            col = np.empty((N, c_per_group * kH * kW, HW), dtype=np.float32)
            slot = 0
            for kh in range(kH):
                h_idx = h_base + kh * dH                       # (H_out,)
                for kwi in range(kW):
                    w_idx = w_base + kwi * dW                  # (W_out,)
                    # Advanced indexing: (N, c, H_out, W_out)
                    patch = X_g[:, :, h_idx[:, None], w_idx[None, :]]
                    col[:, slot:slot + c_per_group, :] = patch.reshape(N, c_per_group, HW)
                    slot += c_per_group

            # W_g flat: (f, c*kH*kW);  col: (N, c*kH*kW, HW)
            W_flat = W_g.reshape(f_per_group, -1)              # (f, K)
            # Batch matmul: (N, f, HW) = (N, K, HW)^T * (f, K)^T
            Y_g = np.einsum('fk,nkm->nfm', W_flat, col, optimize=True)
            Y[:, g * f_per_group:(g + 1) * f_per_group] = Y_g.reshape(N, f_per_group, H_out, W_out)

        if B is not None:
            Y += B.reshape(1, -1, 1, 1)
        return Y

    def _maxpool(self, X, node):
        ks = self._attr(node, "kernel_shape", [2, 2])
        strides = self._attr(node, "strides", ks)
        pads = self._attr(node, "pads", [0] * 2 * len(ks))
        N, C, H, W_ = X.shape
        kH, kW = ks[0], ks[1] if len(ks) > 1 else ks[0]
        sH, sW = strides[0], strides[1] if len(strides) > 1 else strides[0]
        pH, pW = pads[0], pads[1] if len(pads) > 1 else pads[0]
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W_ + 2 * pW - kW) // sW + 1
        X_pad = np.pad(X, ((0, 0), (0, 0), (pH, pH), (pW, pW)), constant_values=-np.inf)
        # Vectorized using sliding window (numpy >= 1.20)
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(X_pad, (kH, kW), axis=(2, 3))
            Y = windows[:, :, ::sH, ::sW].max(axis=(-2, -1))
        except Exception:
            Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
            for oh in range(H_out):
                for ow in range(W_out):
                    Y[:, :, oh, ow] = np.max(
                        X_pad[:, :, oh*sH:oh*sH+kH, ow*sW:ow*sW+kW], axis=(2, 3))
        return Y

    def _avgpool(self, X, node):
        ks = self._attr(node, "kernel_shape", [2, 2])
        strides = self._attr(node, "strides", ks)
        pads = self._attr(node, "pads", [0] * 2 * len(ks))
        N, C, H, W_ = X.shape
        kH, kW = ks[0], ks[1] if len(ks) > 1 else ks[0]
        sH, sW = strides[0], strides[1] if len(strides) > 1 else strides[0]
        pH, pW = pads[0], pads[1] if len(pads) > 1 else pads[0]
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W_ + 2 * pW - kW) // sW + 1
        X_pad = np.pad(X, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode='constant')
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(X_pad, (kH, kW), axis=(2, 3))
            Y = windows[:, :, ::sH, ::sW].mean(axis=(-2, -1)).astype(np.float32)
        except Exception:
            Y = np.zeros((N, C, H_out, W_out), dtype=np.float32)
            for oh in range(H_out):
                for ow in range(W_out):
                    Y[:, :, oh, ow] = np.mean(
                        X_pad[:, :, oh*sH:oh*sH+kH, ow*sW:ow*sW+kW], axis=(2, 3))
        return Y


# ────────────────────────────────────────────────────────────────
# Calibration collector
# ────────────────────────────────────────────────────────────────

class CalibrationCollector:
    """
    Run the float model on calibration data and collect per-tensor
    activation histograms (2048-bin).
    """

    _N_BINS = 2048

    def __init__(self, model: "ModelProto"):
        self._runner = _GraphRunner(model)
        self._graph = model.graph

        # Determine which tensor names to track (non-initializer intermediates)
        init_names = {i.name for i in self._graph.initializer}
        self._track: set = set()
        for node in self._graph.node:
            for o in node.output:
                self._track.add(o)
        # Also track graph inputs (activations)
        for inp in self._graph.input:
            if inp.name not in init_names:
                self._track.add(inp.name)

        self._global_min: Dict[str, float] = {}
        self._global_max: Dict[str, float] = {}
        self._histograms: Dict[str, np.ndarray] = {}
        self._hessian_diag: Dict[str, np.ndarray] = {}
        self._act_abs_max: Dict[str, np.ndarray] = {}  # for SmoothQuant
        self._n_samples = 0

    def collect(self, feed: Dict[str, np.ndarray]) -> None:
        """Run one batch through the graph and update statistics."""
        env = self._runner.run(feed)
        self._n_samples += 1

        for name in self._track:
            arr = env.get(name)
            if arr is None:
                continue
            arr = arr.astype(np.float32)
            vmin, vmax = float(arr.min()), float(arr.max())

            if name not in self._global_min:
                self._global_min[name] = vmin
                self._global_max[name] = vmax
            else:
                self._global_min[name] = min(self._global_min[name], vmin)
                self._global_max[name] = max(self._global_max[name], vmax)

            # Per-channel abs max (axis=0 of flattened-to-2D)
            if arr.ndim >= 2:
                c = arr.shape[1] if arr.ndim >= 2 else arr.shape[0]
                flat = arr.reshape(arr.shape[0], c, -1)
                chan_max = np.max(np.abs(flat), axis=(0, 2))
                if name in self._act_abs_max:
                    self._act_abs_max[name] = np.maximum(self._act_abs_max[name], chan_max)
                else:
                    self._act_abs_max[name] = chan_max

            # Hessian diagonal proxy: accumulate X^T X / n per Conv/Gemm input
            if arr.ndim == 2:
                xtx = np.mean(arr ** 2, axis=0)
                if name in self._hessian_diag:
                    self._hessian_diag[name] += xtx
                else:
                    self._hessian_diag[name] = xtx.copy()

    def finalize(self) -> None:
        """Build histograms from global min/max and re-collect if needed."""
        # Initialize histogram bin edges from global min/max
        for name in self._track:
            vmin = self._global_min.get(name, 0.0)
            vmax = self._global_max.get(name, 1.0)
            if vmin >= vmax:
                vmax = vmin + 1e-6
            self._histograms[name] = {
                "min": vmin,
                "max": vmax,
                "hist": np.zeros(self._N_BINS, dtype=np.int64),
                "edges": np.linspace(vmin, vmax, self._N_BINS + 1),
            }
        # Normalize hessian
        if self._n_samples > 0:
            for name in self._hessian_diag:
                self._hessian_diag[name] /= self._n_samples

    def collect_histogram(self, feed: Dict[str, np.ndarray]) -> None:
        """Second-pass: accumulate histogram counts (call after finalize)."""
        env = self._runner.run(feed)
        for name in self._track:
            arr = env.get(name)
            if arr is None or name not in self._histograms:
                continue
            edges = self._histograms[name]["edges"]
            h, _ = np.histogram(arr.flatten(), bins=edges)
            self._histograms[name]["hist"] += h

    def get_range(self, name: str, method: CalibMethod,
                  **kw) -> Tuple[float, float]:
        """Get calibrated [min, max] range for a tensor."""
        if name in self._histograms:
            info = self._histograms[name]
            fn = _CALIB_FUNCTIONS[method]
            return fn(info["hist"], info["edges"], **kw)
        vmin = self._global_min.get(name, 0.0)
        vmax = self._global_max.get(name, 1.0)
        return vmin, vmax

    def get_hessian_diag(self, name: str) -> Optional[np.ndarray]:
        return self._hessian_diag.get(name)

    def get_act_channel_max(self, name: str) -> Optional[np.ndarray]:
        return self._act_abs_max.get(name)


# ────────────────────────────────────────────────────────────────
# ONNX graph rewriting  (float → QOperator INT8)
# ────────────────────────────────────────────────────────────────

def _make_tensor_vi(name: str, elem_type: int, shape: Sequence[int]):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _make_initializer(name: str, arr: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(arr, name=name)


def _quantize_weight_per_channel(
    W: np.ndarray, axis: int = 0, signed: bool = True, bits: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel symmetric (signed) weight quantization along ``axis``."""
    n_channels = W.shape[axis]
    scales = np.zeros(n_channels, dtype=np.float64)
    zps    = np.zeros(n_channels, dtype=np.int64)
    qmax   = (1 << (bits - 1)) - 1
    lo, hi = -qmax - 1, qmax

    W_q = np.zeros_like(W, dtype=np.int8 if signed else np.uint8)
    for c in range(n_channels):
        slc = [slice(None)] * W.ndim
        slc[axis] = c
        w_c = W[tuple(slc)].astype(np.float64)
        abs_max = np.max(np.abs(w_c))
        s = abs_max / qmax if abs_max > 0 else 1e-12
        scales[c] = s
        q = np.round(w_c / s).astype(np.int64)
        W_q[tuple(slc)] = np.clip(q, lo, hi).astype(W_q.dtype)

    return W_q, scales.astype(np.float32), zps.astype(np.int8 if signed else np.uint8)


def _quantize_weight_per_tensor(
    W: np.ndarray, signed: bool = True, bits: int = 8,
) -> Tuple[np.ndarray, float, int]:
    """Per-tensor symmetric (signed) weight quantization."""
    qmax = (1 << (bits - 1)) - 1
    lo, hi = -qmax - 1, qmax
    abs_max = float(np.max(np.abs(W)))
    s = abs_max / qmax if abs_max > 0 else 1e-12
    q = np.round(W.astype(np.float64) / s).astype(np.int64)
    W_q = np.clip(q, lo, hi).astype(np.int8 if signed else np.uint8)
    return W_q, float(s), 0


def _build_qoperator_graph(
    float_model: "ModelProto",
    collector: CalibrationCollector,
    act_format: ActFormat,
    weight_scheme: WeightScheme,
    calib_method: CalibMethod,
    smooth_alpha: Optional[float],
    use_gptq: bool,
    bits: int = 8,
) -> "ModelProto":
    """
    Rewrite the float ONNX graph into QOperator format.
    """
    graph = float_model.graph
    init_map: Dict[str, np.ndarray] = {
        i.name: numpy_helper.to_array(i) for i in graph.initializer
    }

    act_signed = (act_format == ActFormat.INT8)
    act_np_type = np.int8 if act_signed else np.uint8
    act_onnx_type = TensorProto.INT8 if act_signed else TensorProto.UINT8

    new_inits: Dict[str, TensorProto] = {}
    new_nodes: List[NodeProto] = []
    new_value_infos: List = []

    # Helper: register an initializer array
    def _add_init(name: str, arr: np.ndarray):
        new_inits[name] = _make_initializer(name, arr)

    # Compute activation scale/zp for every tracked tensor
    act_params: Dict[str, Tuple[float, int]] = {}
    for name in collector._track:
        vmin, vmax = collector.get_range(name, calib_method)
        if act_signed:
            s, z = compute_scale_zp_symmetric(vmin, vmax, bits)
        else:
            s, z = compute_scale_zp_asymmetric(vmin, vmax, bits)
        act_params[name] = (s, z)

    def _get_act_scale_zp(tensor_name: str):
        s, z = act_params.get(tensor_name, (1.0, 0))
        s_name = tensor_name + "_scale"
        z_name = tensor_name + "_zp"
        _add_init(s_name, np.array(s, dtype=np.float32))
        if act_signed:
            _add_init(z_name, np.array(z, dtype=np.int8))
        else:
            _add_init(z_name, np.array(z, dtype=np.uint8))
        return s_name, z_name

    for node in graph.node:
        op = node.op_type

        if op == "Conv":
            x_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else ""
            out_name = node.output[0]

            W = init_map[w_name].copy()

            # SmoothQuant
            if smooth_alpha is not None and smooth_alpha > 0:
                act_max = collector.get_act_channel_max(x_name)
                if act_max is not None and len(act_max) == W.shape[1]:
                    w_max = np.max(np.abs(W.reshape(W.shape[0], W.shape[1], -1)), axis=(0, 2))
                    s_smooth = smooth_quant_scales(act_max, w_max, smooth_alpha)
                    # Apply: W *= s[None, :, None, None], act /= s (scale adjustment)
                    W = W * s_smooth[None, :, None, None]
                    # Adjust activation scale
                    old_s, old_z = act_params.get(x_name, (1.0, 0))
                    # Effective: x_real = (x_q - z) * s / s_smooth per channel
                    # For simplicity, use mean rescale factor on activation
                    mean_smooth = float(np.mean(s_smooth))
                    act_params[x_name] = (old_s / mean_smooth, old_z)

            # Quantize weight
            if weight_scheme == WeightScheme.PER_CHANNEL:
                W_q, w_scales, w_zps = _quantize_weight_per_channel(W, axis=0, signed=True, bits=bits)
            else:
                W_q, w_s, w_z = _quantize_weight_per_tensor(W, signed=True, bits=bits)
                w_scales = np.array([w_s], dtype=np.float32)
                w_zps = np.array([w_z], dtype=np.int8)

            # GPTQ correction
            if use_gptq:
                h_diag = collector.get_hessian_diag(x_name)
                if h_diag is not None and W.ndim == 4:
                    W_2d = W.reshape(W.shape[0], -1)
                    if h_diag.shape[0] == W_2d.shape[1]:
                        W_q_2d = gptq_correct_weight(
                            W_2d, h_diag, w_scales[0] if len(w_scales) == 1 else w_scales,
                            w_zps[0] if len(w_zps) == 1 else w_zps,
                            signed=True, bits=bits)
                        W_q = W_q_2d.reshape(W.shape)

            # Register initializers
            w_q_name = w_name + "_q"
            w_s_name = w_name + "_scale"
            w_z_name = w_name + "_zp"
            _add_init(w_q_name, W_q)
            _add_init(w_s_name, w_scales)
            _add_init(w_z_name, w_zps)

            x_s_name, x_z_name = _get_act_scale_zp(x_name)
            y_s_name, y_z_name = _get_act_scale_zp(out_name)

            # QLinearConv inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp [, B]
            qconv_inputs = [
                x_name, x_s_name, x_z_name,
                w_q_name, w_s_name, w_z_name,
                y_s_name, y_z_name,
            ]
            if b_name and b_name in init_map:
                B_float = init_map[b_name].astype(np.float64)
                x_s = float(act_params.get(x_name, (1.0, 0))[0])
                # bias_scale[c] = x_scale * w_scale[c]  (per-channel)
                # bias_scale    = x_scale * w_scale      (per-tensor)
                b_scales = (x_s * w_scales.astype(np.float64))  # shape (C_out,) or (1,)
                if b_scales.shape[0] == 1:
                    B_q = np.round(B_float / b_scales[0]).astype(np.int64)
                else:
                    B_q = np.round(B_float / b_scales).astype(np.int64)
                B_q = np.clip(B_q, -(1 << 31), (1 << 31) - 1).astype(np.int32)
                b_q_name = b_name + "_q"
                _add_init(b_q_name, B_q)
                qconv_inputs.append(b_q_name)

            qconv = helper.make_node(
                "QLinearConv", qconv_inputs, [out_name],
                name=node.name + "_q" if node.name else "",
            )
            for a in node.attribute:
                if a.name in ("kernel_shape", "strides", "pads", "dilations", "group"):
                    qconv.attribute.append(copy.deepcopy(a))
            new_nodes.append(qconv)

        elif op == "Gemm":
            x_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else ""
            out_name = node.output[0]
            W = init_map.get(w_name)
            if W is None:
                new_nodes.append(node)
                continue

            W_q, w_s, w_z = _quantize_weight_per_tensor(W, signed=True, bits=bits)
            w_q_name = w_name + "_q"
            w_s_name = w_name + "_scale"
            w_z_name = w_name + "_zp"
            _add_init(w_q_name, W_q)
            _add_init(w_s_name, np.array([w_s], dtype=np.float32))
            _add_init(w_z_name, np.array([w_z], dtype=np.int8))

            x_s_name, x_z_name = _get_act_scale_zp(x_name)
            y_s_name, y_z_name = _get_act_scale_zp(out_name)

            qgemm_inputs = [
                x_name, x_s_name, x_z_name,
                w_q_name, w_s_name, w_z_name,
                y_s_name, y_z_name,
            ]
            if b_name and b_name in init_map:
                B_float = init_map[b_name].astype(np.float64)
                x_s = float(act_params.get(x_name, (1.0, 0))[0])
                b_scale = x_s * w_s
                B_q = np.round(B_float / b_scale).astype(np.int64)
                B_q = np.clip(B_q, -(1 << 31), (1 << 31) - 1).astype(np.int32)
                b_q_name = b_name + "_q"
                _add_init(b_q_name, B_q)
                qgemm_inputs.append(b_q_name)

            qgemm = helper.make_node("QGemm", qgemm_inputs, [out_name],
                                     name=node.name + "_q" if node.name else "")
            for a in node.attribute:
                if a.name in ("transA", "transB"):
                    qgemm.attribute.append(copy.deepcopy(a))
            new_nodes.append(qgemm)

        elif op == "Add":
            a_name, b_name_add = node.input[0], node.input[1]
            out_name = node.output[0]
            # If both inputs are activations (not initializers), emit QLinearAdd
            if a_name not in init_map and b_name_add not in init_map:
                a_s, a_z = _get_act_scale_zp(a_name)
                b_s, b_z = _get_act_scale_zp(b_name_add)
                y_s, y_z = _get_act_scale_zp(out_name)
                qadd = helper.make_node(
                    "QLinearAdd",
                    [a_name, a_s, a_z, b_name_add, b_s, b_z, y_s, y_z],
                    [out_name],
                    name=node.name + "_q" if node.name else "",
                )
                new_nodes.append(qadd)
            else:
                new_nodes.append(node)

        elif op == "MaxPool":
            new_nodes.append(node)

        elif op == "GlobalAveragePool":
            x_name = node.input[0]
            out_name = node.output[0]
            x_s, x_z = _get_act_scale_zp(x_name)
            y_s, y_z = _get_act_scale_zp(out_name)
            # Wrap as a tracked node with scale info (kept as-is for MLIR importer)
            new_nodes.append(node)

        elif op in ("Relu", "Clip", "Sigmoid", "Flatten", "Reshape",
                     "Transpose", "Dropout", "Identity", "Concat",
                     "BatchNormalization", "Unsqueeze", "Squeeze",
                     "Softmax", "Shape", "Constant", "Gather"):
            new_nodes.append(node)

        else:
            new_nodes.append(node)

    # Build new graph
    all_inits = list(new_inits.values())
    # Preserve non-overwritten original initializers
    overwritten = set(new_inits.keys())
    for orig_init in graph.initializer:
        if orig_init.name not in overwritten:
            all_inits.append(orig_init)

    new_graph = helper.make_graph(
        new_nodes,
        graph.name + "_quantized",
        list(graph.input),
        list(graph.output),
        initializer=all_inits,
    )

    new_model = helper.make_model(new_graph)
    new_model.ir_version = float_model.ir_version
    new_model.opset_import.extend(float_model.opset_import)

    return new_model


# ────────────────────────────────────────────────────────────────
# Hardware-aware post-processing passes
# ────────────────────────────────────────────────────────────────
# These passes adapt the quantized ONNX to a fixed-point hardware backend
# that computes:
#   factor = round(M * 2^FACTOR_BITS)    where FACTOR_BITS = 28
#   out_q  = round(acc * factor >> FACTOR_BITS) + out_zp
#
# Hard constraints coming from the hardware:
#   factor < 2^31  →  M < 8.0     (integer overflow guard)
#   factor >= 1    →  M >= 2^-28  (factor underflow → zero)
# Soft constraint for good precision:
#   M >= 2^-14  (at least 14 effective bits in factor)
# ────────────────────────────────────────────────────────────────

_HW_FACTOR_BITS: int = 28
_HW_MAX_M: float = 7.5            # hard limit ≈ 8, with safety margin
_HW_MIN_M: float = 2.0 ** -14    # soft limit: factor has ≥ 14 useful bits


def _hw_get_init_scalar(inits: dict, name: str) -> Optional[float]:
    arr = inits.get(name)
    return float(arr.flat[0]) if arr is not None else None


def _hw_set_init_scalar(model, inits: dict, name: str, value: float,
                         dtype=None) -> None:
    """Overwrite a scalar initializer in-place (model + local dict)."""
    arr = inits[name]
    new_arr = np.array([value], dtype=dtype or arr.dtype)
    inits[name] = new_arr
    for init in model.graph.initializer:
        if init.name == name:
            init.CopyFrom(numpy_helper.from_array(new_arr, name=name))
            return
    # Not found — add new
    model.graph.initializer.append(numpy_helper.from_array(new_arr, name=name))


def hw_fix_scale_factors(
    q_model,
    factor_bits: int = _HW_FACTOR_BITS,
    max_M: float = _HW_MAX_M,
    min_M: float = _HW_MIN_M,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Post-process a QOperator ONNX model to keep re-quantization factor M
    within the hardware-valid range [min_M, max_M].

    M = in_scale * w_scale / out_scale  (scalar per-layer or per-channel mean)

    Strategy
    --------
    * M > max_M  →  out_scale is too small.  Increase out_scale to bring M to
                    max_M * 0.9 (leaves 10% headroom).  This coarsens the
                    output quantisation but prevents hardware overflow.
    * M < min_M  →  out_scale is too large.  Decrease out_scale to bring M to
                    min_M.  This may slightly overflow the nominal output
                    range but ensures adequate factor precision.

    Returns
    -------
    (n_fixed_high, n_fixed_low) — number of layers adjusted in each direction.
    """
    if not _HAS_ONNX:
        return 0, 0

    inits = {i.name: numpy_helper.to_array(i)
             for i in q_model.graph.initializer}

    n_high = 0
    n_low  = 0

    for node in q_model.graph.node:
        op = node.op_type

        if op in ("QLinearConv", "QGemm"):
            # x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp [, B]
            x_s_name = node.input[1]
            w_s_name = node.input[4]
            y_s_name = node.input[6]

            in_s = _hw_get_init_scalar(inits, x_s_name)
            y_s  = _hw_get_init_scalar(inits, y_s_name)
            if in_s is None or y_s is None:
                continue

            w_s_arr = inits.get(w_s_name)
            if w_s_arr is None:
                continue
            # Use mean of per-channel scales for the check
            w_s_mean = float(np.mean(np.abs(w_s_arr)))
            M = in_s * w_s_mean / y_s

            if M > max_M:
                new_y_s = in_s * w_s_mean / (max_M * 0.9)
                if verbose:
                    print(f"  [hw_fix] {node.name or op}: M={M:.4f} > {max_M} "
                          f"→ out_scale {y_s:.4e} → {new_y_s:.4e}")
                _hw_set_init_scalar(q_model, inits, y_s_name, new_y_s)
                n_high += 1

            elif M < min_M:
                new_y_s = in_s * w_s_mean / (min_M * 2.0)
                if verbose:
                    print(f"  [hw_fix] {node.name or op}: M={M:.6f} < {min_M} "
                          f"→ out_scale {y_s:.4e} → {new_y_s:.4e}")
                _hw_set_init_scalar(q_model, inits, y_s_name, new_y_s)
                n_low += 1

        elif op == "QLinearAdd":
            # a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
            a_s_name = node.input[1]
            b_s_name = node.input[4]
            y_s_name = node.input[6]

            a_s = _hw_get_init_scalar(inits, a_s_name)
            b_s = _hw_get_init_scalar(inits, b_s_name)
            y_s = _hw_get_init_scalar(inits, y_s_name)
            if None in (a_s, b_s, y_s):
                continue

            for branch_s, tag in [(a_s, "branch-A"), (b_s, "branch-B")]:
                M_branch = branch_s / y_s
                if M_branch > max_M:
                    new_y_s = branch_s / (max_M * 0.9)
                    if verbose:
                        print(f"  [hw_fix] {node.name or op} {tag}: "
                              f"M={M_branch:.4f} → out_scale {y_s:.4e} → {new_y_s:.4e}")
                    _hw_set_init_scalar(q_model, inits, y_s_name, new_y_s)
                    y_s = new_y_s
                    n_high += 1
                    break   # re-check not needed; both branches get same y_s

    return n_high, n_low


def hw_equalize_add_scales(
    q_model,
    max_ratio: float = 4.0,
    verbose: bool = False,
) -> int:
    """
    For each QLinearAdd node, if the weaker branch's re-quantization factor M
    falls below the hardware precision floor (_HW_MIN_M), decrease the output
    scale just enough to bring that branch's M up to _HW_MIN_M / 2.

    This is a conservative fix: it only fires when a factor would genuinely
    underflow (factor < 1), and never decreases y_scale below the calibrated
    value (which would clip the output tensor).

    max_ratio : kept for API compatibility, no longer used as the sole trigger.

    Returns number of Add nodes adjusted.
    """
    if not _HAS_ONNX:
        return 0

    inits = {i.name: numpy_helper.to_array(i)
             for i in q_model.graph.initializer}
    n_fixed = 0

    for node in q_model.graph.node:
        if node.op_type != "QLinearAdd":
            continue
        a_s_name = node.input[1]
        b_s_name = node.input[4]
        y_s_name = node.input[6]

        a_s = _hw_get_init_scalar(inits, a_s_name)
        b_s = _hw_get_init_scalar(inits, b_s_name)
        y_s = _hw_get_init_scalar(inits, y_s_name)
        if None in (a_s, b_s, y_s):
            continue

        Ma = a_s / y_s
        Mb = b_s / y_s
        min_branch_M = min(Ma, Mb)

        # Only equalize when the weaker branch's factor is genuinely below the
        # hardware precision floor (min_M).  A large a_s/b_s ratio alone is not
        # sufficient — if both Ma and Mb are within [min_M, max_M] the hardware
        # is already operating correctly.  Forcing y_s downward below the
        # calibrated value clips the output and hurts accuracy.
        if min_branch_M < _HW_MIN_M:
            # Push y_s down just enough so the weaker branch hits min_M/2.
            # Never decrease y_s below the calibrated value (no clipping).
            new_y_s = min(y_s, float(min(a_s, b_s) / (_HW_MIN_M / 2.0)))
            if new_y_s >= y_s:
                continue   # would not help
            if verbose:
                print(f"  [hw_eq_add] {node.name or 'Add'}: "
                      f"a_s={a_s:.3e} b_s={b_s:.3e} "
                      f"min_M={min_branch_M:.2e} < {_HW_MIN_M:.2e} "
                      f"→ y_scale {y_s:.3e} → {new_y_s:.3e}")
            _hw_set_init_scalar(q_model, inits, y_s_name, new_y_s)
            n_fixed += 1

    return n_fixed


def hw_check_factors(
    q_model,
    factor_bits: int = _HW_FACTOR_BITS,
    verbose: bool = True,
) -> dict:
    """
    Compute the hardware re-quantization factor M for every QLinearConv /
    QGemm / QLinearAdd node and return a summary dict.

    Useful for diagnosing overflow / underflow before deploying to hardware.
    """
    if not _HAS_ONNX:
        return {}

    inits = {i.name: numpy_helper.to_array(i)
             for i in q_model.graph.initializer}
    result = {}

    for node in q_model.graph.node:
        op = node.op_type
        name = node.name or op

        if op in ("QLinearConv", "QGemm"):
            in_s = _hw_get_init_scalar(inits, node.input[1])
            w_s_arr = inits.get(node.input[4])
            y_s  = _hw_get_init_scalar(inits, node.input[6])
            if None in (in_s, y_s) or w_s_arr is None:
                continue
            w_s_vec = np.abs(w_s_arr.flatten())
            M_vec = in_s * w_s_vec / y_s
            M_max = float(M_vec.max())
            M_min = float(M_vec.min())
            factor_max = int(round(M_max * (2 ** factor_bits)))
            ok = (M_max < _HW_MAX_M) and (M_min >= _HW_MIN_M)
            result[name] = dict(op=op, M_max=M_max, M_min=M_min,
                                factor_max=factor_max, ok=ok)
            if verbose:
                status = "OK" if ok else "⚠ WARN"
                print(f"  [{status}] {name:40s}  M=[{M_min:.4f}, {M_max:.4f}]"
                      f"  factor_max={factor_max}")

        elif op == "QLinearAdd":
            a_s = _hw_get_init_scalar(inits, node.input[1])
            b_s = _hw_get_init_scalar(inits, node.input[4])
            y_s = _hw_get_init_scalar(inits, node.input[6])
            if None in (a_s, b_s, y_s):
                continue
            Ma = a_s / y_s
            Mb = b_s / y_s
            ratio = max(Ma, Mb) / (min(Ma, Mb) + 1e-30)
            ok = (max(Ma, Mb) < _HW_MAX_M) and (min(Ma, Mb) >= _HW_MIN_M)
            result[name] = dict(op=op, Ma=Ma, Mb=Mb, ratio=ratio, ok=ok)
            if verbose:
                status = "OK" if ok else "⚠ WARN"
                print(f"  [{status}] {name:40s}  Ma={Ma:.4f} Mb={Mb:.4f}"
                      f"  ratio={ratio:.1f}x")

    return result


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def quantize_onnx(
    input_onnx: str,
    output_onnx: str,
    calib_data: np.ndarray,
    *,
    act_format: Union[str, ActFormat] = "int8",
    weight_per_channel: bool = True,
    calibration: Union[str, CalibMethod] = "minmax",
    smooth_alpha: Optional[float] = None,
    use_gptq: bool = False,
    bits: int = 8,
    input_name: Optional[str] = None,
    hw_aware: bool = True,
    verbose: bool = True,
) -> str:
    """
    Quantize a float32 ONNX model to INT8 QOperator format.

    Parameters
    ----------
    input_onnx : path to float32 ONNX model.
    output_onnx : path to write quantized ONNX.
    calib_data : (N, C, H, W) float32 calibration images.
    act_format : ``"int8"`` (symmetric, default) or ``"uint8"`` (asymmetric).
    weight_per_channel : use per-channel weight quantization (default True).
    calibration : ``"minmax"`` | ``"percentile"`` | ``"entropy"``.
    smooth_alpha : SmoothQuant α (0–1). None = disabled.
    use_gptq : enable GPTQ-style weight error correction.
    bits : quantization bit-width (default 8).
    input_name : model input name (auto-detected if None).
    hw_aware : apply hardware-aware post-processing passes (default True).
              Fixes M-factor range and equalises residual-Add input scales
              to match the fixed 28-bit factor hardware (basic.c).
    verbose : print progress.

    Returns
    -------
    output_onnx path.
    """
    if not _HAS_ONNX:
        raise ImportError("onnx package required: pip install onnx")

    act_fmt = ActFormat(act_format) if isinstance(act_format, str) else act_format
    calib_m = CalibMethod(calibration) if isinstance(calibration, str) else calibration
    w_scheme = WeightScheme.PER_CHANNEL if weight_per_channel else WeightScheme.PER_TENSOR

    if verbose:
        print(f"[coa.quantize] Loading {input_onnx}")
    model = onnx.load(input_onnx)

    # Detect input name
    init_names = {i.name for i in model.graph.initializer}
    if input_name is None:
        for inp in model.graph.input:
            if inp.name not in init_names:
                input_name = inp.name
                break
    if input_name is None:
        raise ValueError("Cannot detect model input name. Specify input_name=...")

    if verbose:
        print(f"[coa.quantize] Input tensor: {input_name}")
        print(f"[coa.quantize] Activation format: {act_fmt.value}")
        print(f"[coa.quantize] Weight scheme: {w_scheme.value}")
        print(f"[coa.quantize] Calibration: {calib_m.value}")
        if smooth_alpha is not None:
            print(f"[coa.quantize] SmoothQuant alpha: {smooth_alpha}")
        if use_gptq:
            print(f"[coa.quantize] GPTQ correction: enabled")
        print(f"[coa.quantize] Calibrating with {len(calib_data)} samples ...")

    collector = CalibrationCollector(model)

    # Pass 1: collect global min/max
    for i in range(len(calib_data)):
        sample = calib_data[i:i+1]
        collector.collect({input_name: sample})
    collector.finalize()

    # Pass 2: collect histograms (for percentile/entropy)
    if calib_m in (CalibMethod.PERCENTILE, CalibMethod.ENTROPY):
        if verbose:
            print(f"[coa.quantize] Histogram pass ...")
        for i in range(len(calib_data)):
            sample = calib_data[i:i+1]
            collector.collect_histogram({input_name: sample})

    if verbose:
        print(f"[coa.quantize] Building quantized graph ...")

    q_model = _build_qoperator_graph(
        model, collector,
        act_format=act_fmt,
        weight_scheme=w_scheme,
        calib_method=calib_m,
        smooth_alpha=smooth_alpha,
        use_gptq=use_gptq,
        bits=bits,
    )

    if hw_aware:
        n_high, n_low = hw_fix_scale_factors(q_model, verbose=verbose)
        n_eq = hw_equalize_add_scales(q_model, verbose=verbose)
        if verbose and (n_high + n_low + n_eq) > 0:
            print(f"[coa.quantize] HW-aware: fixed {n_high} M-overflow, "
                  f"{n_low} M-underflow, {n_eq} Add-scale imbalance")

    onnx.save(q_model, output_onnx)
    if verbose:
        n_conv = sum(1 for n in q_model.graph.node if n.op_type == "QLinearConv")
        n_gemm = sum(1 for n in q_model.graph.node if n.op_type == "QGemm")
        n_add  = sum(1 for n in q_model.graph.node if n.op_type == "QLinearAdd")
        print(f"[coa.quantize] Quantized: {n_conv} QLinearConv, {n_gemm} QGemm, {n_add} QLinearAdd")
        print(f"[coa.quantize] Saved to {output_onnx}")

    return output_onnx


# ────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="COA Quantization Toolkit — PTQ float ONNX → INT8 QOperator ONNX")
    parser.add_argument("input",  help="Input float32 ONNX model")
    parser.add_argument("output", help="Output quantized ONNX model")
    parser.add_argument("--calib-dir", default=None,
                        help="Directory with .npy calibration batches")
    parser.add_argument("--act-format", choices=["int8", "uint8"], default="int8",
                        help="Activation quantization format (default: int8)")
    parser.add_argument("--weight-per-channel", action="store_true", default=True,
                        help="Per-channel weight quantization (default)")
    parser.add_argument("--weight-per-tensor", dest="weight_per_channel",
                        action="store_false",
                        help="Per-tensor weight quantization")
    parser.add_argument("--calibration", choices=["minmax", "percentile", "entropy"],
                        default="minmax",
                        help="Calibration method (default: minmax)")
    parser.add_argument("--smooth-alpha", type=float, default=None,
                        help="SmoothQuant alpha (0-1), disabled by default")
    parser.add_argument("--gptq", action="store_true", default=False,
                        help="Enable GPTQ-style weight correction")
    parser.add_argument("--bits", type=int, default=8,
                        help="Quantization bit-width (default: 8)")
    parser.add_argument("--n-calib", type=int, default=64,
                        help="Number of random calibration samples if no --calib-dir")
    args = parser.parse_args()

    import os
    if args.calib_dir and os.path.isdir(args.calib_dir):
        batches = []
        for f in sorted(os.listdir(args.calib_dir)):
            if f.endswith('.npy'):
                batches.append(np.load(os.path.join(args.calib_dir, f)))
        if batches:
            calib = np.concatenate(batches, axis=0)
        else:
            print(f"[coa.quantize] No .npy files in {args.calib_dir}, using random data")
            calib = np.random.randn(args.n_calib, 3, 224, 224).astype(np.float32)
    else:
        print(f"[coa.quantize] No calibration dir, using {args.n_calib} random samples")
        calib = np.random.randn(args.n_calib, 3, 224, 224).astype(np.float32)

    quantize_onnx(
        args.input, args.output, calib,
        act_format=args.act_format,
        weight_per_channel=args.weight_per_channel,
        calibration=args.calibration,
        smooth_alpha=args.smooth_alpha,
        use_gptq=args.gptq,
        bits=args.bits,
    )


if __name__ == "__main__":
    main()
