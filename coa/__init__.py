# coa — Python compiler utilities
# Direct Python ports of the C++ COA compiler internals.
#
#   coa.tiling        — buffer-constraint checker + greedy tile search
#   coa.vliw          — 512-bit VLIW instruction bit-packing
#   coa.mlir_parser   — COA MLIR → layer parameter dicts
#   coa.onnx_importer — quantized ONNX → Level-1 COA MLIR text
#   coa.quantize      — PTQ quantization toolkit (float ONNX → INT8 QOperator ONNX)
#   coa.hw_export     — hardware parameter extraction (quantized ONNX → weight/bias/LUT/factor)
#   coa.pruning       — structural pruning (float ONNX → pruned float ONNX)
from .tiling import calculate_buffer_consumption, get_tile, buffer_utilization
from .vliw import VLIW
from .mlir_parser import parse_all_layers
from .quantize import (
    quantize_onnx,
    hw_fix_scale_factors,
    hw_equalize_add_scales,
    hw_check_factors,
)
from .hw_export import (
    export_hw_params,
    extract_params,
    build_act_lut,
    ConvParams,
    AddParams,
    ActLUT,
)
from .pruning import prune_onnx

__all__ = [
    "calculate_buffer_consumption",
    "get_tile",
    "buffer_utilization",
    "VLIW",
    "parse_all_layers",
    "quantize_onnx",
    "hw_fix_scale_factors",
    "hw_equalize_add_scales",
    "hw_check_factors",
    "export_hw_params",
    "extract_params",
    "build_act_lut",
    "ConvParams",
    "AddParams",
    "ActLUT",
    "prune_onnx",
]
