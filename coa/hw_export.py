"""
coa.hw_export — Hardware parameter extraction for FPGA deployment.

Reads a QOperator ONNX model (output of ``coa.quantize``) and produces
hardware-ready binary files that match the legacy C-reference format
(``legacy/c_reference/resnet_inference.c``, ``legacy/parameters/``):

    weight.image        INT8 weights   layout: (OC, KH, KW, IC_pad32)
    weight_offset.txt   per-layer address map
    bias.image          INT32 biases   layout: (OC_pad16,)
    bias_offset.txt     per-layer address map
    act.image           256-entry INT8 LUT per activation layer
                        (combines activation + cross-layer re-quantisation)
    act_offset.txt      per-layer LUT address map
    factors.json        per-layer **single** INT factor = floor(M × 2^factor_bits)

Hardware fixed-point convention (operator.h / basic.c):
    Conv/Gemm :  factor = floor(M × 2^36)   (CONV_FACTOR_BITS = 36)
                 out_q  = round(acc × factor / 2^36) + out_zp
    Add/Res   :  factor = floor(M × 2^28)   (ADD_FACTOR_BITS  = 28)
    M         = in_scale × w_scale / out_scale   ← scalar (per-tensor)

Per-channel → per-tensor conversion (automatic when force_per_tensor=True):
    1. Dequant:   w_float = (w_int8 - w_zp_pc) × w_scale_pc
    2. New scale: w_scale_pt = max|w_float| / 127
    3. Re-quant:  w_q_pt = clip(round(w_float / w_scale_pt), -128, 127)
    4. Bias adj:  b_q_pt = round(b_q_pc × w_scale_pc / w_scale_pt)

Activation LUT convention (relu_offset.txt):
    lut[i]  = clip(round(act_fn(dequant(i)) / out_scale) + out_zp, -128, 127)
    where dequant(i) = (int8(i) - in_zp) × in_scale
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

# ── Hardware constants ────────────────────────────────────────────────────────
CONV_FACTOR_BITS: int = 36      # Conv/Gemm: factor = floor(M × 2^36)  (operator.h QLinearConv_AUTO)
ADD_FACTOR_BITS:  int = 28      # Add/Res:   factor = floor(M × 2^28)  (operator.h QLinearAdd)
FACTOR_BITS:      int = CONV_FACTOR_BITS  # default alias
IC_PAD_UNIT: int = 32           # FPGA processes 32 IC-channels per cycle
OC_BIAS_PAD: int = 16           # bias OC count aligned to multiples of 16
LUT_SIZE:    int = 256          # activation LUT has 256 INT8 entries
INT8_LO,     INT8_HI = -128, 127


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvParams:
    """Parameters for one QLinearConv or QGemm node."""
    name:        str
    op:          str             # "QLinearConv" | "QGemm"
    in_scale:    float
    in_zp:       int
    w_int8:      np.ndarray      # original ONNX INT8 weight (OC, IC, KH, KW)
    w_scale:     np.ndarray      # (OC,) float32
    w_zp:        np.ndarray      # (OC,) int32
    bias_int32:  Optional[np.ndarray]  # (OC,) int32, or None
    out_scale:   float
    out_zp:      int
    strides:     Tuple[int, ...]  = field(default_factory=lambda: (1, 1))
    pads:        Tuple[int, ...]  = field(default_factory=lambda: (0, 0, 0, 0))

    # ── derived helpers ───────────────────────────────────────────────────────
    @property
    def OC(self) -> int:
        return self.w_int8.shape[0]

    @property
    def IC(self) -> int:
        return self.w_int8.shape[1] if self.w_int8.ndim >= 2 else 1

    @property
    def KH(self) -> int:
        return self.w_int8.shape[2] if self.w_int8.ndim == 4 else 1

    @property
    def KW(self) -> int:
        return self.w_int8.shape[3] if self.w_int8.ndim == 4 else 1

    def to_per_tensor(self) -> "ConvParams":
        """
        Convert per-channel quantized weights to per-tensor.

        If already per-tensor (w_scale has length 1), returns self unchanged.
        Otherwise:
          1. Dequant:   w_float = (w_int8 - w_zp_pc) × w_scale_pc
          2. New scale: w_scale_pt = max|w_float| / 127
          3. Re-quant:  w_q_pt = clip(round(w_float / w_scale_pt), -128, 127)
          4. Bias adj:  b_q_pt = round(b_q_pc × w_scale_pc / w_scale_pt)
        """
        if self.w_scale.size == 1:
            return self

        # Broadcast w_zp / w_scale over all spatial / IC dims
        # w_int8 shape: (OC, IC, KH, KW) for Conv, (OC, IC) for Gemm
        extra_dims = self.w_int8.ndim - 1   # number of dims after OC
        idx = (slice(None),) + (np.newaxis,) * extra_dims
        w_float = ((self.w_int8.astype(np.float64) - self.w_zp[idx].astype(np.float64))
                   * self.w_scale[idx].astype(np.float64))

        w_scale_pt = float(np.max(np.abs(w_float)) / 127.0)
        if w_scale_pt < 1e-12:
            w_scale_pt = 1e-8

        w_q_pt = np.clip(np.round(w_float / w_scale_pt),
                         INT8_LO, INT8_HI).astype(np.int8)

        # Bias: b_float = b_q_pc * in_scale * w_scale_pc[oc]
        # b_q_pt[oc] = round(b_float / (in_scale * w_scale_pt))
        #            = round(b_q_pc * w_scale_pc / w_scale_pt)
        b_pt = None
        if self.bias_int32 is not None:
            b_pt = np.round(
                self.bias_int32.astype(np.float64)
                * self.w_scale.astype(np.float64) / w_scale_pt
            ).astype(np.int32)

        return replace(
            self,
            w_int8=w_q_pt,
            w_scale=np.array([w_scale_pt], dtype=np.float32),
            w_zp=np.zeros(1, dtype=np.int32),
            bias_int32=b_pt,
        )

    def factor_pt(self, factor_bits: int = CONV_FACTOR_BITS) -> int:
        """
        Single per-tensor factor = floor(M × 2^factor_bits).
        Expects w_scale to be scalar (call to_per_tensor() first).
        """
        w_s = float(self.w_scale.flat[0])
        M   = self.in_scale * w_s / self.out_scale
        return int(np.floor(M * float(2 ** factor_bits)))

    def factors(self) -> np.ndarray:
        """Legacy per-channel factors array (for diagnostics only)."""
        M = self.in_scale * self.w_scale / self.out_scale
        f = np.floor(M * float(2 ** CONV_FACTOR_BITS)).astype(np.int64)
        f = np.clip(f, -(2**63), 2**63 - 1)
        return f

    def weight_hw(self) -> np.ndarray:
        """
        Repack weights to FPGA layout.

        ONNX Conv:  (OC, IC, KH, KW) → transpose → pad IC → (OC, KH, KW, IC_pad32)
        ONNX Gemm:  (OC, IC)         → transpose → pad IC → (IC_pad32, OC)
                    (matches legacy extract_gemm_weight convention)
        """
        w = self.w_int8
        if w.ndim == 2:
            # Gemm: (OC, IC) → (IC, OC) → (IC_pad32, OC)
            w = w.T                         # (IC, OC)
            IC, OC = w.shape
            IC_pad = ((IC + IC_PAD_UNIT - 1) // IC_PAD_UNIT) * IC_PAD_UNIT
            if IC_pad > IC:
                w = np.pad(w, [(0, IC_pad - IC), (0, 0)],
                           mode="constant", constant_values=0)
        else:
            # Conv: (OC, IC, KH, KW) → (OC, KH, KW, IC)
            w = w.transpose(0, 2, 3, 1)
            OC, KH, KW, IC = w.shape
            IC_pad = ((IC + IC_PAD_UNIT - 1) // IC_PAD_UNIT) * IC_PAD_UNIT
            if IC_pad > IC:
                w = np.pad(w, [(0, 0), (0, 0), (0, 0), (0, IC_pad - IC)],
                           mode="constant", constant_values=0)
        return w.astype(np.int8)

    def bias_padded(self) -> np.ndarray:
        """INT32 bias, OC count padded to multiple of OC_BIAS_PAD (zeros if missing)."""
        OC_pad = ((self.OC + OC_BIAS_PAD - 1) // OC_BIAS_PAD) * OC_BIAS_PAD
        if self.bias_int32 is not None:
            b = self.bias_int32.flatten().astype(np.int32)
        else:
            b = np.zeros(self.OC, dtype=np.int32)
        if len(b) < OC_pad:
            b = np.concatenate([b, np.zeros(OC_pad - len(b), dtype=np.int32)])
        return b[:OC_pad]


@dataclass
class AddParams:
    """Parameters for one QLinearAdd node."""
    name:      str
    a_scale:   float
    a_zp:      int
    b_scale:   float
    b_zp:      int
    out_scale: float
    out_zp:    int

    def factor_a(self) -> int:
        return int(np.floor(self.a_scale / self.out_scale * float(2 ** ADD_FACTOR_BITS)))

    def factor_b(self) -> int:
        return int(np.floor(self.b_scale / self.out_scale * float(2 ** ADD_FACTOR_BITS)))


@dataclass
class ActLUT:
    """256-entry INT8 activation lookup table for one layer."""
    name:      str             # layer name this LUT belongs to
    act_type:  str             # "relu" | "clip" | "sigmoid" | "identity" | ...
    in_scale:  float
    in_zp:     int
    out_scale: float
    out_zp:    int
    lut:       np.ndarray      # (256,) int8


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_scalar(inits: dict, name: str) -> Optional[float]:
    arr = inits.get(name)
    return float(arr.flat[0]) if arr is not None else None


def _get_int(inits: dict, name: str, default: int = 0) -> int:
    arr = inits.get(name)
    return int(arr.flat[0]) if arr is not None else default


def _get_array(inits: dict, name: str) -> Optional[np.ndarray]:
    return inits.get(name)


def _attr(node, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return tuple(a.ints)
            if a.floats:
                return tuple(a.floats)
            return a.i or a.f or a.s or default
    return default


# ── Node parsers ─────────────────────────────────────────────────────────────

def _parse_qlinearconv(node, inits: dict) -> Optional[ConvParams]:
    # inputs: x x_s x_zp w w_s w_zp y_s y_zp [B]
    if len(node.input) < 8:
        return None
    in_s  = _get_scalar(inits, node.input[1])
    in_z  = _get_int(inits, node.input[2])
    w_arr = _get_array(inits, node.input[3])
    w_s   = _get_array(inits, node.input[4])
    w_z   = _get_array(inits, node.input[5])
    y_s   = _get_scalar(inits, node.input[6])
    y_z   = _get_int(inits, node.input[7])
    b_arr = (_get_array(inits, node.input[8])
             if len(node.input) > 8 and node.input[8] else None)
    if any(v is None for v in [in_s, w_arr, w_s, y_s]):
        return None
    return ConvParams(
        name=node.name or node.output[0],
        op="QLinearConv",
        in_scale=float(in_s), in_zp=int(in_z),
        w_int8=w_arr.astype(np.int8),
        w_scale=w_s.flatten().astype(np.float32),
        w_zp=(w_z.flatten().astype(np.int32)
              if w_z is not None else np.zeros(w_s.size, np.int32)),
        bias_int32=(b_arr.flatten().astype(np.int32) if b_arr is not None else None),
        out_scale=float(y_s), out_zp=int(y_z),
        strides=_attr(node, "strides", (1, 1)),
        pads=_attr(node, "pads", (0, 0, 0, 0)),
    )


def _parse_qgemm(node, inits: dict) -> Optional[ConvParams]:
    # com.microsoft.QGemm inputs: A A_s A_zp B B_s B_zp C(bias) Y_s Y_zp
    if len(node.input) < 8:
        return None
    in_s  = _get_scalar(inits, node.input[1])
    in_z  = _get_int(inits, node.input[2])
    w_arr = _get_array(inits, node.input[3])
    w_s   = _get_array(inits, node.input[4])
    w_z   = _get_array(inits, node.input[5])
    # input[6] = C (bias, optional); input[7] = Y_scale; input[8] = Y_zero_point
    b_arr = (_get_array(inits, node.input[6])
             if len(node.input) > 6 and node.input[6] else None)
    y_s   = _get_scalar(inits, node.input[7]) if len(node.input) > 7 else None
    y_z   = _get_int(inits, node.input[8])    if len(node.input) > 8 else 0
    if any(v is None for v in [in_s, w_arr, w_s, y_s]):
        return None
    return ConvParams(
        name=node.name or node.output[0],
        op="QGemm",
        in_scale=float(in_s), in_zp=int(in_z),
        w_int8=w_arr.astype(np.int8),
        w_scale=w_s.flatten().astype(np.float32),
        w_zp=(w_z.flatten().astype(np.int32)
              if w_z is not None else np.zeros(w_s.size, np.int32)),
        bias_int32=(b_arr.flatten().astype(np.int32) if b_arr is not None else None),
        out_scale=float(y_s), out_zp=int(y_z),
        strides=(1, 1), pads=(0, 0, 0, 0),
    )


def _parse_qlinearadd(node, inits: dict) -> Optional[AddParams]:
    # inputs: a a_s a_zp b b_s b_zp y_s y_zp
    if len(node.input) < 8:
        return None
    a_s = _get_scalar(inits, node.input[1])
    a_z = _get_int(inits, node.input[2])
    b_s = _get_scalar(inits, node.input[4])
    b_z = _get_int(inits, node.input[5])
    y_s = _get_scalar(inits, node.input[6])
    y_z = _get_int(inits, node.input[7])
    if any(v is None for v in [a_s, b_s, y_s]):
        return None
    return AddParams(
        name=node.name or node.output[0],
        a_scale=float(a_s), a_zp=int(a_z),
        b_scale=float(b_s), b_zp=int(b_z),
        out_scale=float(y_s), out_zp=int(y_z),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Activation LUT generation
# ═══════════════════════════════════════════════════════════════════════════════

def build_act_lut(
    act_type:  str,
    in_scale:  float,
    in_zp:     int,
    out_scale: float,
    out_zp:    int,
    clip_min:  Optional[float] = None,
    clip_max:  Optional[float] = None,
) -> np.ndarray:
    """
    Build a 256-entry INT8 LUT that combines dequant → activation → requant.

        lut[i] = clip(round(act_fn((int8(i) - in_zp) × in_scale) / out_scale)
                      + out_zp, -128, 127)

    act_type : "relu" | "clip" | "sigmoid" | "silu" | "gelu" | "identity"
    clip_min / clip_max : used when act_type == "clip"

    Returns uint8 array of length 256 (raw bytes; hardware indexes with
    the int8 value reinterpreted as uint8).
    """
    indices  = np.arange(256, dtype=np.uint8).view(np.int8)   # -128 … 127
    x_float  = (indices.astype(np.float32) - in_zp) * in_scale

    if act_type == "relu":
        y_float = np.maximum(0.0, x_float)
    elif act_type == "clip":
        lo = clip_min if clip_min is not None else -np.inf
        hi = clip_max if clip_max is not None else np.inf
        y_float = np.clip(x_float, lo, hi)
    elif act_type == "sigmoid":
        y_float = 1.0 / (1.0 + np.exp(-x_float))
    elif act_type == "silu":
        y_float = x_float / (1.0 + np.exp(-x_float))
    elif act_type == "gelu":
        import math
        y_float = (x_float * 0.5 *
                   (1.0 + np.vectorize(math.erf)(x_float / math.sqrt(2))))
    else:   # identity / unknown
        y_float = x_float

    q = np.round(y_float / out_scale) + out_zp
    q = np.clip(q, INT8_LO, INT8_HI).astype(np.int8)
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# Topological sort helper
# ═══════════════════════════════════════════════════════════════════════════════

def _topo_sort_nodes(graph) -> list:
    """
    Return graph.node in BFS topological order (Kahn's algorithm).

    Among nodes that become simultaneously ready (all data-dependencies
    satisfied), the order follows their original position in *graph.node*.
    This is important for residual networks: in a ResNet block the shortcut
    (1×1) conv shares its input with the main-branch first conv, so both
    become "ready" at the same wave.  BFS processes them in original order
    (main-branch conv1 first, shortcut second), then the main-branch conv2
    follows in the next wave — matching the VLIW execution convention.

    Falls back to the original node order if a cycle is detected.
    """
    from collections import defaultdict, deque

    nodes      = list(graph.node)
    always_avail = (
        {inp.name  for inp in graph.input}
        | {init.name for init in graph.initializer}
    )

    # produced_by[tensor] = index of the node that produces it
    produced_by: dict = {}
    for i, nd in enumerate(nodes):
        for out in nd.output:
            if out:
                produced_by[out] = i

    # consumers[tensor] = list of node-indices that consume it as a DATA dep
    consumers: dict = defaultdict(list)
    pending   = [0] * len(nodes)
    for i, nd in enumerate(nodes):
        for inp in nd.input:
            if inp and inp not in always_avail and inp in produced_by:
                consumers[inp].append(i)
                pending[i] += 1

    queue  = deque(i for i, p in enumerate(pending) if p == 0)
    result = []
    while queue:
        i = queue.popleft()
        result.append(nodes[i])
        for out in nodes[i].output:
            if out in consumers:
                for j in consumers[out]:
                    pending[j] -= 1
                    if pending[j] == 0:
                        queue.append(j)

    if len(result) != len(nodes):
        return nodes          # cycle guard – fall back to raw order
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction pass
# ═══════════════════════════════════════════════════════════════════════════════

def extract_params(
    q_model,
) -> Tuple[List[Union[ConvParams, AddParams]], List[ActLUT]]:
    """
    Walk a QOperator ONNX graph and return:
        (conv_add_list, lut_list)

    conv_add_list : ConvParams and AddParams in topological order.
    lut_list      : ActLUT for each activation node in topological order.
    """
    if not _HAS_ONNX:
        raise ImportError("onnx package required")

    inits = {i.name: numpy_helper.to_array(i)
             for i in q_model.graph.initializer}

    # Map: tensor_name → (scale, zp) for scale tracking
    tensor_scale: Dict[str, Tuple[float, int]] = {}

    # Seed from initializer names following "_scale" / "_zero_point" convention
    for iname, arr in inits.items():
        if iname.endswith("_scale"):
            base = iname[: -len("_scale")]
            zp_arr = inits.get(base + "_zero_point")
            zp = int(zp_arr.flat[0]) if zp_arr is not None else 0
            tensor_scale[base] = (float(arr.flat[0]), zp)

    # For activation nodes we need to know the output scale of the NEXT
    # compute node (Conv/Gemm/Add) — build a forward output→next-conv map.
    # We do this with a simple two-pass approach.
    output_to_node: Dict[str, object] = {}
    for node in q_model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    def _next_compute_scale(tensor_name: str) -> Optional[Tuple[float, int]]:
        """Follow the graph forward to find the (scale, zp) of the next
        conv/gemm/add node that consumes this tensor."""
        visited = set()
        queue = [tensor_name]
        while queue:
            t = queue.pop(0)
            if t in visited:
                continue
            visited.add(t)
            # Find node that consumes t
            for node in q_model.graph.node:
                if t not in node.input:
                    continue
                op = node.op_type
                if op in ("QLinearConv", "QGemm"):
                    s = _get_scalar(inits, node.input[1])
                    z = _get_int(inits, node.input[2])
                    if s is not None:
                        return (s, z)
                elif op == "QLinearAdd":
                    # use the add output scale as proxy
                    s = _get_scalar(inits, node.input[6])
                    z = _get_int(inits, node.input[7])
                    if s is not None:
                        return (s, z)
                else:
                    # passthrough: follow outputs
                    queue.extend(node.output)
        return None

    conv_add_list: List[Union[ConvParams, AddParams]] = []
    lut_list:      List[ActLUT] = []

    for node in _topo_sort_nodes(q_model.graph):
        op = node.op_type

        if op == "QLinearConv":
            p = _parse_qlinearconv(node, inits)
            if p:
                conv_add_list.append(p)
                tensor_scale[node.output[0]] = (p.out_scale, p.out_zp)

        elif op == "QGemm":
            p = _parse_qgemm(node, inits)
            if p:
                conv_add_list.append(p)
                tensor_scale[node.output[0]] = (p.out_scale, p.out_zp)

        elif op == "QLinearAdd":
            p = _parse_qlinearadd(node, inits)
            if p:
                conv_add_list.append(p)
                tensor_scale[node.output[0]] = (p.out_scale, p.out_zp)

        elif op in ("Relu", "Clip", "Sigmoid", "Silu", "Gelu"):
            inp_tensor = node.input[0]
            in_sv = tensor_scale.get(inp_tensor)
            if in_sv is None:
                continue
            in_s, in_z = in_sv

            # Output scale: same tensor (passthrough in QOperator) — try
            # to find what the downstream conv uses as its input scale.
            nxt = _next_compute_scale(node.output[0])
            if nxt is None:
                out_s, out_z = in_s, in_z   # no re-quantise needed
            else:
                out_s, out_z = nxt

            # Clip parameters
            clip_min = clip_max = None
            if op == "Clip" and len(node.input) >= 3:
                mn = _get_scalar(inits, node.input[1]) if node.input[1] else None
                mx = _get_scalar(inits, node.input[2]) if node.input[2] else None
                clip_min = float(mn) * in_s if mn is not None else None
                clip_max = float(mx) * in_s if mx is not None else None

            lut = build_act_lut(op.lower(), in_s, in_z, out_s, out_z,
                                clip_min, clip_max)
            lut_list.append(ActLUT(
                name=node.name or f"{op}_{inp_tensor}",
                act_type=op.lower(),
                in_scale=in_s, in_zp=in_z,
                out_scale=out_s, out_zp=out_z,
                lut=lut,
            ))
            tensor_scale[node.output[0]] = (out_s, out_z)

    return conv_add_list, lut_list


# ═══════════════════════════════════════════════════════════════════════════════
# Binary writers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_offset_header(f, title: str) -> None:
    f.write(f"# {title}\n")
    f.write("-" * 70 + "\n")


def _write_weight_image(conv_list, out_dir: str, verbose: bool) -> dict:
    """Write weight.image + weight_offset.txt.  Returns offset dict."""
    img_path = os.path.join(out_dir, "weight.image")
    off_path = os.path.join(out_dir, "weight_offset.txt")
    offsets  = {}
    addr     = 0

    with open(img_path, "wb") as fimg, open(off_path, "w") as foff:
        _write_offset_header(foff, "Weight 偏移地址映射")
        for p in conv_list:
            if not isinstance(p, ConvParams):
                continue
            w_hw = p.weight_hw()          # Conv:(OC,KH,KW,IC_pad32)  Gemm:(IC_pad32,OC)
            raw  = w_hw.tobytes()
            fimg.write(raw)
            if p.op == "QGemm":
                IC_pad, OC = w_hw.shape
                shape_str = f"({IC_pad}, {OC})"
                ic_pad = IC_pad
            else:
                OC, KH, KW, ic_pad = w_hw.shape
                shape_str = f"({OC}, {KH}, {KW}, {ic_pad})"
            line = (f"0x{addr:08x}: {p.op.lower()} {p.name} "
                    f"| weight shape:{shape_str} ic_pad:{ic_pad}\n")
            foff.write(line)
            offsets[p.name] = {"addr": addr, "shape": list(w_hw.shape)}
            addr += len(raw)
            if verbose:
                print(f"  [weight] {p.name:40s}  {shape_str}  @0x{addr-len(raw):08x}")
        foff.write("-" * 70 + "\n")
        foff.write(f"# Total: {addr} bytes\n")

    return offsets


def _write_bias_image(conv_list, out_dir: str, verbose: bool) -> dict:
    """Write bias.image + bias_offset.txt.  Returns offset dict."""
    img_path = os.path.join(out_dir, "bias.image")
    off_path = os.path.join(out_dir, "bias_offset.txt")
    coe_path = os.path.join(out_dir, "bias.coe")
    offsets  = {}
    addr     = 0

    with open(img_path, "wb") as fimg, \
         open(off_path, "w")  as foff, \
         open(coe_path, "w")  as fcoe:
        _write_offset_header(foff, "Bias 偏移地址映射")
        fcoe.write("memory_initialization_radix = 16;\n"
                   "memory_initialization_vector=\n")

        for p in conv_list:
            if not isinstance(p, ConvParams):
                continue
            b = p.bias_padded()          # (OC_pad16,) int32
            raw = b.tobytes()            # little-endian int32
            fimg.write(raw)

            # COE line: each int32 printed as 8 hex digits (MSB first per channel)
            hex_vals = "".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in b)
            fcoe.write(hex_vals + ",\n")

            line = (f"0x{addr:08x}: {p.name} "
                    f"| bias shape:({p.OC}) pad:{len(b)}\n")
            foff.write(line)
            offsets[p.name] = {"addr": addr, "oc": p.OC, "oc_pad": len(b)}
            addr += len(raw)
            if verbose:
                print(f"  [bias]   {p.name:40s}  OC={p.OC} pad={len(b)}"
                      f"  @0x{addr-len(raw):08x}")

        foff.write("-" * 70 + "\n")
        foff.write(f"# Total: {addr} bytes\n")

    return offsets


def _write_act_image(lut_list, out_dir: str, verbose: bool) -> dict:
    """Write act.image + act_offset.txt.  Returns offset dict."""
    img_path = os.path.join(out_dir, "act.image")
    coe_path = os.path.join(out_dir, "act.coe")
    off_path = os.path.join(out_dir, "act_offset.txt")
    offsets  = {}
    addr     = 0

    with open(img_path, "wb") as fimg, \
         open(off_path, "w")  as foff, \
         open(coe_path, "w")  as fcoe:
        _write_offset_header(foff, "Activation LUT 偏移地址映射")
        _write_offset_header(foff, "Format: 地址: layer_idx name | "
                             "in_s=... in_zp=... out_s=... out_zp=...")
        foff.write("-" * 70 + "\n")
        fcoe.write("memory_initialization_radix = 16;\n"
                   "memory_initialization_vector=\n")

        for idx, lut_entry in enumerate(lut_list):
            raw = lut_entry.lut.view(np.uint8).tobytes()
            fimg.write(raw)

            # COE: 16 bytes per line (legacy format)
            lut_u8 = lut_entry.lut.view(np.uint8)
            coe_lines = []
            for row_start in range(0, 256, 16):
                row = lut_u8[row_start: row_start + 16]
                coe_lines.append("".join(f"{b:02x}" for b in row))
            fcoe.write(",\n".join(coe_lines) + ",\n")

            line = (f"0x{addr:08x}: layer_{idx} {lut_entry.name} "
                    f"| in_s={lut_entry.in_scale:.6f} in_zp={lut_entry.in_zp} "
                    f"out_s={lut_entry.out_scale:.6f} out_zp={lut_entry.out_zp}\n")
            foff.write(line)
            offsets[lut_entry.name] = {
                "addr": addr, "act": lut_entry.act_type,
                "in_scale": lut_entry.in_scale,  "in_zp": lut_entry.in_zp,
                "out_scale": lut_entry.out_scale, "out_zp": lut_entry.out_zp,
            }
            addr += len(raw)
            if verbose:
                print(f"  [act]    {lut_entry.name:40s}"
                      f"  {lut_entry.act_type:8s}"
                      f"  in_s={lut_entry.in_scale:.5f}"
                      f"  out_s={lut_entry.out_scale:.5f}")

        foff.write("-" * 70 + "\n")
        foff.write(f"# Total: {addr} bytes ({len(lut_list)} layers x {LUT_SIZE} bytes)\n")

    return offsets


def _write_factors_json(conv_add_list, out_dir: str, verbose: bool) -> dict:
    """
    Write factors.json.  Each Conv/Gemm entry has a **single** integer factor
    (per-tensor).  ConvParams must already be per-tensor (w_scale scalar).
    AddParams entries have two factors (fa, fb), both at ADD_FACTOR_BITS.
    """
    factors_dict: dict = {}

    for p in conv_add_list:
        if isinstance(p, ConvParams):
            f   = p.factor_pt()          # single int
            w_s = float(p.w_scale.flat[0])
            M   = p.in_scale * w_s / p.out_scale
            # Validity: factor must fit in 34-bit signed VLIW field
            ok  = (0 < f < (2**33 - 1))
            entry = {
                "op":          p.op,
                "in_scale":    p.in_scale,
                "w_scale":     w_s,
                "out_scale":   p.out_scale,
                "M":           M,
                "factor":      f,
                "factor_bits": CONV_FACTOR_BITS,
                "ok":          ok,
            }
            if verbose:
                status = "OK" if ok else "⚠ OVF"
                print(f"  [factor] [{status}] {p.name:40s}"
                      f"  M={M:.6f}  factor=0x{f:08x} ({f})")
        else:  # AddParams
            fa = p.factor_a()
            fb = p.factor_b()
            entry = {
                "op":          "QLinearAdd",
                "a_scale":     p.a_scale, "a_zp": p.a_zp,
                "b_scale":     p.b_scale, "b_zp": p.b_zp,
                "out_scale":   p.out_scale,
                "factor_a":    fa,
                "factor_b":    fb,
                "factor_bits": ADD_FACTOR_BITS,
            }
            if verbose:
                print(f"  [factor] {p.name:40s}"
                      f"  fa=0x{fa:08x}  fb=0x{fb:08x}")
        factors_dict[p.name] = entry

    out_path = os.path.join(out_dir, "factors.json")
    with open(out_path, "w") as f:
        json.dump(factors_dict, f, indent=2)

    return factors_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def export_hw_params(
    q_model_or_path,
    output_dir: str,
    force_per_tensor: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Extract all hardware parameters from a quantized QOperator ONNX model
    and write binary blobs + offset maps to *output_dir*.

    Parameters
    ----------
    q_model_or_path  : onnx.ModelProto or str path to .onnx file.
    output_dir       : directory to write output files (created if absent).
    force_per_tensor : if True (default), automatically convert per-channel
                       weight quantization to per-tensor so that each Conv
                       layer has exactly one factor (as required by the VLIW
                       instruction encoding).
    verbose          : print per-layer summary.

    Returns
    -------
    Summary dict with keys:
        "weight_offsets", "bias_offsets", "act_offsets", "factors"
    """
    if not _HAS_ONNX:
        raise ImportError("onnx package required: pip install onnx")

    if isinstance(q_model_or_path, str):
        q_model = onnx.load(q_model_or_path)
    else:
        q_model = q_model_or_path

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"[hw_export] Extracting parameters → {output_dir}")

    conv_add_list, lut_list = extract_params(q_model)

    n_conv = sum(1 for p in conv_add_list if isinstance(p, ConvParams))
    n_add  = sum(1 for p in conv_add_list if isinstance(p, AddParams))

    if verbose:
        print(f"[hw_export] Found {n_conv} Conv/Gemm, {n_add} Add, "
              f"{len(lut_list)} activation LUTs")

    # ── Per-channel → per-tensor conversion ──────────────────────────────────
    if force_per_tensor:
        n_converted = 0
        new_list = []
        for p in conv_add_list:
            if isinstance(p, ConvParams) and p.w_scale.size > 1:
                p = p.to_per_tensor()
                n_converted += 1
            new_list.append(p)
        conv_add_list = new_list
        if verbose and n_converted:
            print(f"[hw_export] Converted {n_converted} layers: "
                  f"per-channel → per-tensor")
    print()

    conv_list = [p for p in conv_add_list if isinstance(p, ConvParams)]

    if verbose:
        print("── Weights ──")
    w_off  = _write_weight_image(conv_list, output_dir, verbose)

    if verbose:
        print("\n── Biases ──")
    b_off  = _write_bias_image(conv_list, output_dir, verbose)

    if verbose:
        print("\n── Activation LUTs ──")
    a_off  = _write_act_image(lut_list, output_dir, verbose)

    if verbose:
        print("\n── Factors ──")
    factors = _write_factors_json(conv_add_list, output_dir, verbose)

    summary = {
        "weight_offsets": w_off,
        "bias_offsets":   b_off,
        "act_offsets":    a_off,
        "factors":        factors,
    }
    summary_path = os.path.join(output_dir, "hw_params_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        total_w = sum(
            int(np.prod(v["shape"])) for v in w_off.values()
        )
        total_b = sum(v["oc_pad"] * 4 for v in b_off.values())
        total_a = len(a_off) * LUT_SIZE
        print(f"\n[hw_export] weight.image  : {total_w:>10,} bytes")
        print(f"[hw_export] bias.image    : {total_b:>10,} bytes")
        print(f"[hw_export] act.image     : {total_a:>10,} bytes")
        print(f"[hw_export] factors.json  : {len(factors):>10} layers")
        print(f"[hw_export] All files written to {output_dir}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract FPGA hardware parameters from quantized ONNX.\n"
                    "Default output: examples/resnet18/parameters/")
    ap.add_argument("input_onnx",  help="quantized QOperator ONNX path")
    ap.add_argument("output_dir",
                    nargs="?", default=None,
                    help="output directory (default: examples/<model>/parameters/ "
                         "next to the ONNX file)")
    ap.add_argument("--no-per-tensor", action="store_true",
                    help="skip per-channel → per-tensor weight conversion")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.output_dir is None:
        base = os.path.splitext(os.path.basename(args.input_onnx))[0]
        args.output_dir = os.path.join(
            os.path.dirname(args.input_onnx), "..", "parameters", base)

    export_hw_params(args.input_onnx, args.output_dir,
                     force_per_tensor=not args.no_per_tensor,
                     verbose=not args.quiet)
