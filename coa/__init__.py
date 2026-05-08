# coa — Python compiler utilities
# Direct Python ports of the C++ COA compiler internals.
#
#   coa.tiling      — buffer-constraint checker + greedy tile search
#   coa.vliw        — 512-bit VLIW instruction bit-packing
#   coa.mlir_parser — COA MLIR → layer parameter dicts
#   coa.onnx_importer — quantized ONNX → Level-1 COA MLIR text
from .tiling import calculate_buffer_consumption, get_tile, buffer_utilization
from .vliw import VLIW
from .mlir_parser import parse_all_layers

__all__ = [
    "calculate_buffer_consumption",
    "get_tile",
    "buffer_utilization",
    "VLIW",
    "parse_all_layers",
]
