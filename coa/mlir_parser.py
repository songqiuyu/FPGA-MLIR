"""
coa.mlir_parser — Parse a (fully-lowered) COA MLIR file and extract per-layer
parameter dicts that are directly compatible with VLIW(**layer).

Python port of legacy/tools/extract_vliw.py.

Supported ops:
  coa.qlinearconv              → operator=0
  coa.qgemm                    → operator=0  (GEMM uses Conv encoding)
  coa.maxpool                  → operator=1
  coa.qlinearglobalaveragepool → operator=2
  coa.qlinearadd               → operator=3
"""

import re
from typing import Any, Dict, List, Optional

_OP_CODES: Dict[str, int] = {
    "qlinearconv":              0,
    "qgemm":                    0,
    "maxpool":                  1,
    "qlinearglobalaveragepool": 2,
    "qlinearadd":               3,
}


def _parse_int(s: str, default: int = 0) -> int:
    try:
        return int(s.strip(), 0)
    except (ValueError, TypeError):
        return default


def _parse_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return default


def _attr_int(attrs: str, name: str, default: int = 0) -> int:
    """Extract a single integer attribute, e.g.  R = 112."""
    m = re.search(
        rf'\b{re.escape(name)}\s*=\s*(0x[0-9a-fA-F]+|-?\d+)',
        attrs
    )
    return _parse_int(m.group(1)) if m else default


def _attr_list_first_int(attrs: str, name: str, default: int = 0) -> int:
    """Extract first element of an integer list attribute, e.g.  kernel_shape = [7, 7]."""
    m = re.search(rf'\b{re.escape(name)}\s*=\s*\[([^\]]*)\]', attrs)
    if not m:
        return default
    nums = re.findall(r'-?\d+', m.group(1))
    return int(nums[0]) if nums else default


def parse_all_layers(mlir_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse a COA MLIR file and return a list of layer parameter dicts.

    Each dict is compatible with ``VLIW(**layer)`` — field names match the
    VLIW constructor's keyword arguments.  An extra ``'type'`` key holds the
    lower-case op name (e.g. ``'qlinearconv'``).

    Only fully-lowered MLIR (with tM/tR/tC and *_addr attributes) will
    produce complete VLIW dicts.  Level-1 MLIR (before --coa-tiling /
    --coa-addr-assign) will have zero for those fields.
    """
    with open(mlir_path, "r", encoding="utf-8") as f:
        text = f.read()

    layers: List[Dict[str, Any]] = []

    # Match each "coa.<opname>"(...) { ... } block.
    # The attribute block ends at the *first* unmatched '}'.
    op_re = re.compile(
        r'"coa\.(\w+)"\s*\([^)]*\)\s*\{([^}]*)\}',
        re.DOTALL,
    )

    for m in op_re.finditer(text):
        op_name = m.group(1).lower()
        attrs   = m.group(2)

        operator = _OP_CODES.get(op_name, 0)

        def _i(name: str, default: int = 0) -> int:
            return _attr_int(attrs, name, default)

        def _li(name: str, default: int = 0) -> int:
            return _attr_list_first_int(attrs, name, default)

        R = _i("R")
        C = _i("C")
        M = _i("M")
        N = _i("N")

        layer: Dict[str, Any] = {
            "type":                         op_name,
            "operator":                     operator,
            "DDR_x1_address":               _i("in_addr"),
            "DDR_x2_address":               _i("weight_addr"),
            "Bias_source_address":          _i("bias_addr") & 0x7FF,
            "Compute_Result_dest_address":  _i("out_addr"),
            "Activate_LUT_address":         (_i("silu_addr") >> 24) & 0xFF,
            "R":                            R,
            "C":                            C,
            "M":                            M,
            "N":                            N,
            "R0":                           _i("R0", R),
            "C0":                           _i("C0", C),
            "sM_concat":                    _i("sM_concat", M),
            "M_concat":                     _i("M_concat",  M),
            "Quant_x1_z":                   _i("in_zp"),
            "Quant_x2_z":                   _li("weight_zp"),
            "Quant_y_z":                    _i("out_zp"),
            "Conv_pad":                     _li("pads"),
            "Conv_kernel":                  _li("kernel_shape", 1),
            "Conv_stride":                  _li("strides", 1),
            "Conv_dilation":                _li("dilations", 1),
            "Conv_tR":                      _i("tR", R),
            "Conv_tC":                      _i("tC", C),
            "Conv_tM":                      _i("tM", M),
            "Conv_tN":                      N,
            "Conv_permuteR":                0,
            "Conv_permuteC":                0,
            "Conv_permuteM":                0,
            "Conv_permuteN":                0,
            "Conv_quant_factor":            _i("factor"),
            "Conv_quant_factor2":           _i("factor2"),
        }

        if verbose:
            print(f"[mlir_parser] {op_name}: "
                  f"R={layer['R']} C={layer['C']} "
                  f"M={layer['M']} N={layer['N']}")
        layers.append(layer)

    return layers
