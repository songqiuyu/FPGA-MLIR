"""
AutoQuant - Mixed Precision Quantization Application (Phase 3-C)
Applies a bit-assignment config (from pareto_search.py) to a COA MLIR file,
generating a new MLIR with per-layer quantization scale/factor overrides.

Usage:
  python mixed_quant.py --config mixed_precision_config.json \
                        --mlir   model_assigned.mlir \
                        --output model_mixed.mlir
"""

import argparse
import json
import re
import os
import sys
import math
from typing import Dict

import numpy as np


def _requantize_scale(original_scale: float, bits: int) -> float:
    """
    Adjust quantization scale for a different bit-width.
    For symmetric uniform quant: scale = max_val / (2^(bits-1) - 1)
    We approximate by rescaling the original 8-bit scale.
    """
    ref_levels  = 2 ** (8 - 1) - 1   # 127 for int8
    new_levels  = 2 ** (bits - 1) - 1
    return original_scale * (ref_levels / new_levels)


def _requantize_factor(factor: int, bits: int) -> int:
    """
    Adjust the fixed-point quant factor for a different bit-width.
    factor = round(in_scale * w_scale / out_scale * 2^15)
    If bit-width changes, scale changes proportionally.
    """
    ref_levels = 2 ** (8 - 1) - 1
    new_levels = 2 ** (bits - 1) - 1
    ratio = (new_levels / ref_levels) ** 2  # both input and weight scale change
    return int(round(factor * ratio))


def apply_mixed_precision(mlir_text: str,
                           bit_assignment: Dict[str, int],
                           weight_names_map: Dict[str, str] = None) -> str:
    """
    Rewrite a COA MLIR string with per-layer bit-width overrides.

    Args:
        mlir_text:        Original COA MLIR text (output of assign_addr pass)
        bit_assignment:   {weight_tensor_name: n_bits}  from AutoQuant
        weight_names_map: Optional {mlir_weight_var: canonical_weight_name}

    Returns:
        Modified MLIR text with updated in_scale, out_scale, factor values.
    """
    lines_out = []
    op_pattern = re.compile(
        r'(%\w+)\s*=\s*"coa\.(qlinearconv|qgemm)"\(([^)]*)\)\s*\{([^}]*)\}'
    )

    for line in mlir_text.splitlines():
        m = op_pattern.search(line)
        if not m:
            lines_out.append(line)
            continue

        out_var  = m.group(1)
        op_type  = m.group(2)
        args_str = m.group(3)
        attrs    = m.group(4)

        # Identify the weight variable name
        arg_list = [a.strip().lstrip("%") for a in args_str.split(",")]
        weight_var = arg_list[1] if len(arg_list) > 1 else ""

        # Look up the bit-width for this layer
        bits = bit_assignment.get(weight_var, None)
        if weight_names_map:
            canonical = weight_names_map.get(weight_var, weight_var)
            bits = bit_assignment.get(canonical, bits)

        if bits is None or bits == 8:
            lines_out.append(line)
            continue

        # Adjust in_scale / out_scale / factor
        def replace_float_attr(text, name, new_val):
            return re.sub(
                rf'(\b{name}\s*=\s*)([\d.eE+\-]+)',
                lambda mo: f"{mo.group(1)}{new_val:.8f}",
                text
            )

        def replace_int_attr(text, name, new_val):
            return re.sub(
                rf'(\b{name}\s*=\s*)(0x[0-9a-fA-F]+|\d+)',
                lambda mo: f"{mo.group(1)}{new_val}",
                text
            )

        # Parse current values
        in_s_m  = re.search(r'\bin_scale\s*=\s*([\d.eE+\-]+)', attrs)
        out_s_m = re.search(r'\bout_scale\s*=\s*([\d.eE+\-]+)', attrs)
        fac_m   = re.search(r'\bfactor\s*=\s*(\d+)', attrs)

        new_attrs = attrs
        if in_s_m:
            orig_in_s  = float(in_s_m.group(1))
            new_attrs = replace_float_attr(new_attrs, "in_scale",
                                           _requantize_scale(orig_in_s, bits))
        if out_s_m:
            orig_out_s = float(out_s_m.group(1))
            new_attrs = replace_float_attr(new_attrs, "out_scale",
                                           _requantize_scale(orig_out_s, bits))
        if fac_m:
            orig_fac = int(fac_m.group(1))
            new_attrs = replace_int_attr(new_attrs, "factor",
                                         _requantize_factor(orig_fac, bits))

        # Reconstruct the line
        new_line = line[:m.start(4)] + new_attrs + line[m.end(4):]
        # Append a comment indicating the bit-width override
        new_line = new_line.rstrip() + f"  // mixed-prec: {bits}-bit"
        lines_out.append(new_line)

    return "\n".join(lines_out)


def main():
    parser = argparse.ArgumentParser(
        description="Apply mixed-precision quantization config to COA MLIR")
    parser.add_argument("--config",  required=True,
                        help="Path to mixed_precision_config.json")
    parser.add_argument("--mlir",    required=True,
                        help="Input COA MLIR file (after assign-addr pass)")
    parser.add_argument("--output",  required=True,
                        help="Output COA MLIR file with mixed precision")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    bit_assignment = config.get("bit_assignment", {})

    with open(args.mlir, encoding="utf-8") as f:
        mlir_text = f.read()

    avg_bits = config.get("avg_bits", "?")
    print(f"[mixed_quant] Applying mixed precision: avg {avg_bits:.2f} bits/param")
    print(f"  Layers with non-8-bit: "
          f"{sum(1 for b in bit_assignment.values() if b != 8)}/{len(bit_assignment)}")

    out_text = apply_mixed_precision(mlir_text, bit_assignment)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"[mixed_quant] Output written to {args.output}")


if __name__ == "__main__":
    main()
