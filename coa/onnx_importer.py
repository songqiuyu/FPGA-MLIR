"""
coa.onnx_importer — Convert a quantized INT8 ONNX model to Level-1 COA MLIR.

Python port of legacy/coa_mlir/frontend/{importer.py, mlir_gen.py}.

Supported ONNX ops (onnxruntime QOperator quantization format):
  QLinearConv  → coa.qlinearconv
  MaxPool      → coa.maxpool
  QGemm        → coa.qgemm
  QLinearAdd   → coa.qlinearadd
  GlobalAveragePool / QLinearAveragePool → coa.qlinearglobalaveragepool

Level-1 attributes written (shape-infer / tiling / addr-assign are C++ passes):
  in_scale, in_zp, weight_scale, weight_zp, out_scale, out_zp
  kernel_shape, strides, pads, dilations
  (R, C, M, N filled only when static shape is available from the value info)

Usage:
  from coa.onnx_importer import onnx_to_coa_mlir
  mlir_text = onnx_to_coa_mlir("resnet18_quant_int8.onnx")
  with open("resnet18.mlir", "w") as f:
      f.write(mlir_text)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import onnx
    from onnx import numpy_helper, TensorProto
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def onnx_to_coa_mlir(onnx_path: str, func_name: str = "main") -> str:
    """
    Load a quantized ONNX model and return Level-1 COA MLIR text.

    Raises:
        ImportError  — if the ``onnx`` package is not installed.
        ValueError   — if no supported ops are found in the model.
    """
    if not _HAS_ONNX:
        raise ImportError("onnx package is required: pip install onnx")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    graph = model.graph

    # Build initializer map: name → numpy array
    inits: Dict[str, object] = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }

    # Build value-info shape map: name → (batch, C, H, W) or None
    shapes: Dict[str, Optional[Tuple]] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            dims = tuple(
                d.dim_value if d.HasField("dim_value") else -1
                for d in t.shape.dim
            )
            shapes[vi.name] = dims
        else:
            shapes[vi.name] = None

    # Determine model input/output SSA names
    graph_inputs  = [inp.name for inp in graph.input
                     if inp.name not in inits]
    graph_outputs = [out.name for out in graph.output]

    # Collect layer IR nodes
    layers: List[_Layer] = []
    for node in graph.node:
        layer = _dispatch_node(node, inits, shapes)
        if layer is not None:
            layers.append(layer)

    if not layers:
        raise ValueError(
            f"No supported COA ops found in {onnx_path}. "
            "Supported: QLinearConv, MaxPool, QGemm, QLinearAdd, "
            "GlobalAveragePool."
        )

    return _emit_mlir(layers, graph_inputs, graph_outputs,
                      shapes, func_name, inits=inits)


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

class _Layer:
    """Intermediate representation for one MLIR op."""
    def __init__(self, op: str, result_name: str, inputs: List[str],
                 attrs: Dict, type_str: str):
        self.op          = op           # e.g. "coa.qlinearconv"
        self.result_name = result_name  # SSA value name, e.g. "%conv1"
        self.inputs      = inputs       # SSA value names (activations only)
        self.attrs       = attrs        # dict of MLIR attribute key → value
        self.type_str    = type_str     # full MLIR type signature string


# ---------------------------------------------------------------------------
# Node dispatchers
# ---------------------------------------------------------------------------

_COUNTER: Dict[str, int] = {}

def _fresh_name(prefix: str) -> str:
    _COUNTER[prefix] = _COUNTER.get(prefix, 0) + 1
    return f"%{prefix}{_COUNTER[prefix]}"


def _scalar(inits: Dict, name: str, default: float = 0.0) -> float:
    if name in inits:
        arr = inits[name]
        return float(arr.flat[0])
    return default


def _array(inits: Dict, name: str) -> List[float]:
    if name in inits:
        return inits[name].flatten().tolist()
    return [0.0]


def _shape4(shapes: Dict, name: str) -> Tuple[int, int, int, int]:
    """Return (N_batch, C, H, W) from shape map, 0 for unknown dims."""
    s = shapes.get(name)
    if s and len(s) == 4:
        return tuple(max(d, 0) for d in s)
    return (0, 0, 0, 0)


def _dispatch_node(node, inits: Dict, shapes: Dict) -> Optional[_Layer]:
    op = node.op_type
    if op == "QLinearConv":
        return _parse_qlinearconv(node, inits, shapes)
    if op == "MaxPool":
        return _parse_maxpool(node, inits, shapes)
    if op in ("GlobalAveragePool", "QLinearAveragePool"):
        return _parse_gap(node, inits, shapes)
    if op == "QLinearAdd":
        return _parse_qlinearadd(node, inits, shapes)
    if op in ("QGemm", "QLinearMatMul"):
        return _parse_qgemm(node, inits, shapes)
    return None


def _get_attr(node, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == 7:                          # INTS
                v = list(a.ints)
                return v if v else default
            if a.type == 6:                          # FLOATS
                v = list(a.floats)
                return v if v else default
            if a.type == 1:    return a.f            # FLOAT
            if a.type == 2:    return a.i            # INT
            if a.type == 3:    return a.s            # STRING
    return default


def _parse_qlinearconv(node, inits, shapes):
    # inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp [, B]
    inp = node.input
    x_name   = inp[0]
    w_name   = inp[3] if len(inp) > 3 else ""
    b_name   = inp[8] if len(inp) > 8 else ""

    in_scale = _scalar(inits, inp[1])
    in_zp    = int(_scalar(inits, inp[2]))
    w_scale  = _array(inits, inp[4])
    w_zp     = [int(z) for z in _array(inits, inp[5])]
    out_scale= _scalar(inits, inp[6])
    out_zp   = int(_scalar(inits, inp[7]))

    strides  = _get_attr(node, "strides",   [1, 1])
    pads     = _get_attr(node, "pads",       [0, 0, 0, 0])
    dilations= _get_attr(node, "dilations",  [1, 1])
    group    = _get_attr(node, "group",      1)

    # kernel_shape: prefer explicit attribute, else infer from weight tensor
    kernel = _get_attr(node, "kernel_shape", None)
    if not kernel:
        w_arr = inits.get(w_name)
        if w_arr is not None and w_arr.ndim == 4:
            kernel = list(w_arr.shape[2:])
        else:
            w_shape = shapes.get(w_name)
            kernel = list(w_shape[2:]) if w_shape and len(w_shape) >= 4 else [1, 1]

    out_name = node.output[0]
    _, N, R0, C0 = _shape4(shapes, x_name)
    _, M, R,  C  = _shape4(shapes, out_name)

    result = _fresh_name("conv")
    ssas = [f"%{_ssa(x_name)}", f"%{_ssa(w_name)}", f"%{_ssa(b_name)}"]
    in_type  = _tensor_type(shapes, x_name)
    w_type   = _tensor_type(shapes, w_name)
    b_type   = _bias_type(shapes, b_name, M)
    out_type = _tensor_type(shapes, out_name)

    attrs = {
        "in_scale":    f"{in_scale:.8f} : f64",
        "in_zp":       f"{in_zp} : i32",
        "weight_scale": _fmt_f64_list(w_scale),
        "weight_zp":    _fmt_i64_list(w_zp),
        "out_scale":   f"{out_scale:.8f} : f64",
        "out_zp":      f"{out_zp} : i32",
        "kernel_shape": _fmt_i64_list(kernel),
        "strides":      _fmt_i64_list(strides),
        "pads":         _fmt_i64_list(pads),
        "dilations":    _fmt_i64_list(dilations),
    }
    if R:  attrs["R"]  = str(R)
    if C:  attrs["C"]  = str(C)
    if M:  attrs["M"]  = str(M)
    if N:  attrs["N"]  = str(N)
    if R0: attrs["R0"] = str(R0)
    if C0: attrs["C0"] = str(C0)
    if M:
        attrs["sM_concat"] = str(M)
        attrs["M_concat"]  = str(M)

    type_str = f"({in_type}, {w_type}, {b_type}) -> {out_type}"
    return _Layer("coa.qlinearconv", result, ssas, attrs, type_str)


def _parse_maxpool(node, inits, shapes):
    x_name  = node.input[0]
    out_name= node.output[0]

    in_scale  = 0.0   # MaxPool is pass-through for quant params
    in_zp     = 0
    out_scale = 0.0
    out_zp    = 0

    kernel  = _get_attr(node, "kernel_shape", [2, 2])
    strides = _get_attr(node, "strides",       [1, 1])
    pads    = _get_attr(node, "pads",           [0, 0, 0, 0])

    _, N, R0, C0 = _shape4(shapes, x_name)
    _, M, R, C   = _shape4(shapes, out_name)

    result = _fresh_name("pool")
    attrs = {
        "in_scale":    f"{in_scale:.8f} : f64",
        "in_zp":       f"{in_zp} : i32",
        "out_scale":   f"{out_scale:.8f} : f64",
        "out_zp":      f"{out_zp} : i32",
        "kernel_shape": _fmt_i64_list(kernel),
        "strides":      _fmt_i64_list(strides),
        "pads":         _fmt_i64_list(pads),
    }
    if R:  attrs["R"]  = str(R)
    if C:  attrs["C"]  = str(C)
    if M:  attrs["M"]  = str(M)
    if N:  attrs["N"]  = str(N)
    if R0: attrs["R0"] = str(R0)
    if C0: attrs["C0"] = str(C0)

    in_type  = _tensor_type(shapes, x_name)
    out_type = _tensor_type(shapes, out_name)
    type_str = f"({in_type}) -> {out_type}"
    return _Layer("coa.maxpool", result, [f"%{_ssa(x_name)}"], attrs, type_str)


def _parse_gap(node, inits, shapes):
    x_name   = node.input[0]
    out_name = node.output[0]
    inp      = node.input

    if node.op_type == "QLinearAveragePool":
        in_scale  = _scalar(inits, inp[1]) if len(inp) > 1 else 0.0
        in_zp     = int(_scalar(inits, inp[2])) if len(inp) > 2 else 0
        out_scale = _scalar(inits, inp[3]) if len(inp) > 3 else 0.0
        out_zp    = int(_scalar(inits, inp[4])) if len(inp) > 4 else 0
    else:
        in_scale = out_scale = 0.0
        in_zp = out_zp = 0

    _, N, R0, C0 = _shape4(shapes, x_name)
    _, M, _, _   = _shape4(shapes, out_name)

    result = _fresh_name("gap")
    attrs = {
        "in_scale":  f"{in_scale:.8f} : f64",
        "in_zp":     f"{in_zp} : i32",
        "out_scale": f"{out_scale:.8f} : f64",
        "out_zp":    f"{out_zp} : i32",
    }
    if M:  attrs["M"]  = str(M)
    if N:  attrs["N"]  = str(N)
    if R0: attrs["R0"] = str(R0)
    if C0: attrs["C0"] = str(C0)

    in_type  = _tensor_type(shapes, x_name)
    out_type = _tensor_type(shapes, out_name)
    type_str = f"({in_type}) -> {out_type}"
    return _Layer("coa.qlinearglobalaveragepool", result,
                  [f"%{_ssa(x_name)}"], attrs, type_str)


def _parse_qlinearadd(node, inits, shapes):
    inp = node.input
    a_name   = inp[0]
    b_name   = inp[3] if len(inp) > 3 else inp[1]

    a_scale  = _scalar(inits, inp[1])
    a_zp     = int(_scalar(inits, inp[2]))
    b_scale  = _scalar(inits, inp[4]) if len(inp) > 4 else 0.0
    b_zp     = int(_scalar(inits, inp[5])) if len(inp) > 5 else 0
    out_scale= _scalar(inits, inp[6]) if len(inp) > 6 else 0.0
    out_zp   = int(_scalar(inits, inp[7])) if len(inp) > 7 else 0

    out_name = node.output[0]
    _, M, R, C = _shape4(shapes, out_name)

    result = _fresh_name("add")
    attrs = {
        "a_scale":   f"{a_scale:.8f} : f64",
        "a_zp":      f"{a_zp} : i32",
        "b_scale":   f"{b_scale:.8f} : f64",
        "b_zp":      f"{b_zp} : i32",
        "out_scale": f"{out_scale:.8f} : f64",
        "out_zp":    f"{out_zp} : i32",
    }
    if R: attrs["R"] = str(R)
    if C: attrs["C"] = str(C)
    if M: attrs["M"] = str(M)

    in_type  = _tensor_type(shapes, a_name)
    out_type = _tensor_type(shapes, out_name)
    type_str = f"({in_type}, {in_type}) -> {out_type}"
    return _Layer("coa.qlinearadd", result,
                  [f"%{_ssa(a_name)}", f"%{_ssa(b_name)}"], attrs, type_str)


def _parse_qgemm(node, inits, shapes):
    inp = node.input
    x_name = inp[0]
    w_name = inp[3] if len(inp) > 3 else ""
    b_name = inp[8] if len(inp) > 8 else ""

    in_scale  = _scalar(inits, inp[1])
    in_zp     = int(_scalar(inits, inp[2]))
    w_scale   = _array(inits, inp[4])
    w_zp      = [int(z) for z in _array(inits, inp[5])]
    out_scale = _scalar(inits, inp[6])
    out_zp    = int(_scalar(inits, inp[7]))

    out_name = node.output[0]
    _, N, _, _ = _shape4(shapes, x_name)
    _, M, _, _ = _shape4(shapes, out_name)

    result = _fresh_name("fc")
    ssas = [f"%{_ssa(x_name)}", f"%{_ssa(w_name)}", f"%{_ssa(b_name)}"]
    w_type   = _tensor_type(shapes, w_name)
    b_type   = _bias_type(shapes, b_name, M)
    in_type  = _tensor_type(shapes, x_name)
    out_type = _tensor_type(shapes, out_name)

    attrs = {
        "a_scale":  f"{in_scale:.8f} : f64",
        "a_zp":     f"{in_zp} : i32",
        "b_scale":  _fmt_f64_list(w_scale),
        "b_zp":     _fmt_i64_list(w_zp),
        "out_scale": f"{out_scale:.8f} : f64",
        "out_zp":    f"{out_zp} : i32",
    }
    if M: attrs["M"] = str(M)
    if N: attrs["N"] = str(N)

    type_str = f"({in_type}, {w_type}, {b_type}) -> {out_type}"
    return _Layer("coa.qgemm", result, ssas, attrs, type_str)


# ---------------------------------------------------------------------------
# MLIR text emitter
# ---------------------------------------------------------------------------

def _emit_mlir(layers: List[_Layer],
               graph_inputs: List[str],
               graph_outputs: List[str],
               shapes: Dict,
               func_name: str,
               inits: Dict = None) -> str:
    if inits is None:
        inits = {}
    lines: List[str] = []
    lines.append("module {")

    # --- Determine which SSA names are defined (data inputs + op results) ---
    defined_ssas: set = set(f"%{_ssa(n)}" for n in graph_inputs)

    # Build ssa→type map from each layer's type_str, so weight/bias get correct types.
    ssa_type: Dict[str, str] = {}
    for layer in layers:
        # type_str format: "(t0, t1, t2) -> tout"  or  "t0 -> tout"
        sig = layer.type_str.split("->")[0].strip().strip("()")
        input_types = [t.strip() for t in sig.split(",") if t.strip()]
        for inp, typ in zip(layer.inputs, input_types):
            if inp not in ssa_type:
                ssa_type[inp] = typ

    # Scan ops to find undeclared inputs (weight/bias initializers)
    extra_ssa: List[str] = []   # ordered, unique
    seen_extra: set = set()
    for layer in layers:
        for inp in layer.inputs:
            if inp not in defined_ssas and inp not in seen_extra:
                extra_ssa.append(inp)
                seen_extra.add(inp)
        defined_ssas.add(layer.result_name)

    # Build function argument list: data inputs first, then weight/bias tensors
    arg_parts: List[str] = [
        f"%{_ssa(n)}: {_tensor_type(shapes, n)}" for n in graph_inputs
    ]
    for ssa in extra_ssa:
        arg_type = ssa_type.get(ssa, "tensor<?xi8>")
        arg_parts.append(f"{ssa}: {arg_type}")

    # Use dynamic shapes for function output (shape-infer pass fills them later)
    out_types = ", ".join("tensor<?xi8>" for _ in graph_outputs) if graph_outputs else "tensor<?xi8>"
    args_str = ", ".join(arg_parts)
    lines.append(f"  func.func @{func_name}({args_str}) -> {out_types} {{")

    for layer in layers:
        attr_lines = []
        for k, v in layer.attrs.items():
            attr_lines.append(f"      {k} = {v}")
        attr_str = ",\n".join(attr_lines)

        inputs_str = ", ".join(layer.inputs)
        lines.append(
            f"    {layer.result_name} = \"{layer.op}\"({inputs_str}) {{\n"
            f"{attr_str}\n"
            f"    }} : {layer.type_str}"
        )

    # Return statement
    ret_vals  = layers[-1].result_name if layers else "%result"
    ret_types = ", ".join("tensor<?xi8>" for _ in graph_outputs) if graph_outputs else "tensor<?xi8>"
    lines.append(f"    return {ret_vals} : {ret_types}")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    TensorProto.INT8:   "i8",
    TensorProto.UINT8:  "ui8",
    TensorProto.INT32:  "i32",
    TensorProto.FLOAT:  "f32",
    TensorProto.FLOAT16:"f16",
    TensorProto.INT64:  "i64",
} if _HAS_ONNX else {}


def _tensor_type(shapes: Dict, name: str) -> str:
    s = shapes.get(name)
    if s:
        dims = "x".join(str(max(d, 1)) for d in s)
        return f"tensor<{dims}xi8>"
    return "tensor<?xi8>"


def _bias_type(shapes: Dict, name: str, M: int) -> str:
    s = shapes.get(name)
    if s and len(s) >= 1:
        return f"tensor<{max(s[0], 1)}xi32>"
    if M:
        return f"tensor<{M}xi32>"
    return "tensor<?xi32>"


def _ssa(name: str) -> str:
    """Convert an ONNX tensor name to a legal MLIR SSA identifier."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name).lstrip('_') or "v"


def _fmt_f64_list(values: List[float]) -> str:
    inner = ", ".join(f"{v:.6f} : f64" for v in values)
    return f"[{inner}]"


def _fmt_i32_list(values: List[int]) -> str:
    inner = ", ".join(f"{v} : i32" for v in values)
    return f"[{inner}]"


def _fmt_i64_list(values: List[int]) -> str:
    inner = ", ".join(f"{v} : i64" for v in values)
    return f"[{inner}]"


def _fmt_int_list(values: Sequence) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"
