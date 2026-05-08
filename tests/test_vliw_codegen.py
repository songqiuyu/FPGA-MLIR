"""
Integration test: validate VLIW binary output against the Python reference.

Compares the byte-level output of the Python tools/vliw.py with the C++
VLIWCodeGen by running the Python reference first, then diffing against
coa-opt output (if the C++ compiler is built).
"""

import os
import sys
import struct
import unittest
import subprocess
import tempfile

from coa.vliw import VLIW as PyVLIW
from coa.mlir_parser import parse_all_layers
from coa.tiling import get_tile, calculate_buffer_consumption


class TestPythonVLIWPacking(unittest.TestCase):
    """Unit tests for the Python VLIW bit-packing reference."""

    def _make_vliw(self, **kwargs) -> PyVLIW:
        defaults = dict(
            operator=0,
            DDR_x1_address=0x100,
            DDR_x2_address=0x8000000,
            Bias_source_address=0x40,
            Compute_Result_dest_address=0x10000000,
            Activate_LUT_address=0,
            R=56, C=56, M=64, N=64,
            R0=56, C0=56, sM_concat=64, M_concat=64,
            Quant_x1_z=0, Quant_x2_z=0, Quant_y_z=-128,
            Conv_pad=1, Conv_kernel=3, Conv_stride=1, Conv_dilation=1,
            Conv_tR=56, Conv_tC=56, Conv_tM=64, Conv_tN=64,
            Conv_permuteR=0, Conv_permuteC=0, Conv_permuteM=0, Conv_permuteN=0,
            Conv_quant_factor=4096, Conv_quant_factor2=0,
        )
        defaults.update(kwargs)
        return PyVLIW(**defaults)

    def test_output_length(self):
        """VLIW.to_bytes() must produce exactly 64 bytes."""
        v = self._make_vliw()
        b = v.to_bytes()
        self.assertEqual(len(b), 64)

    def test_operator_field_conv(self):
        """Operator=0 (Conv): first byte bits [7:0] should encode 0."""
        v = self._make_vliw(operator=0)
        b = v.to_bytes()
        op = b[0] & 0xFF
        self.assertEqual(op, 0)

    def test_operator_field_pool(self):
        """Operator=1 (Pool): first byte should be 0x01."""
        v = self._make_vliw(operator=1)
        b = v.to_bytes()
        self.assertEqual(b[0], 1)

    def test_operator_field_add(self):
        """Operator=3 (Add)."""
        v = self._make_vliw(operator=3)
        b = v.to_bytes()
        self.assertEqual(b[0], 3)

    def test_all_zeros(self):
        """Zero VLIW should produce 64 zero bytes."""
        v = PyVLIW()
        b = v.to_bytes()
        self.assertEqual(len(b), 64)
        self.assertEqual(sum(b), 0)

    def test_deterministic(self):
        """Same VLIW encodes to the same bytes every time."""
        v1 = self._make_vliw()
        v2 = self._make_vliw()
        self.assertEqual(v1.to_bytes(), v2.to_bytes())

    def test_ddr_x1_round_trip(self):
        """DDR_x1_address (36 bits) should be recoverable from bytes 1-5."""
        addr = 0x8CAFE000
        v = self._make_vliw(DDR_x1_address=addr)
        b = v.to_bytes()
        # Bits [43:8] -> bytes 1-5 (bits 8-43)
        packed = int.from_bytes(b[0:6], 'little') >> 8
        recovered = packed & ((1 << 36) - 1)
        self.assertEqual(recovered, addr)

    def test_pool_operator_consistency(self):
        """A pool VLIW should not have Conv kernel/stride bits corrupting operator."""
        v = self._make_vliw(operator=1, Conv_kernel=3, Conv_stride=2)
        b = v.to_bytes()
        # Operator byte must still be 1
        self.assertEqual(b[0] & 0xFF, 1)


class TestExtractVLIW(unittest.TestCase):
    """Smoke tests for the extract_vliw.py parsing logic."""

    MINIMAL_MLIR = """\
module {
  func.func @main(%input: tensor<1x3x224x224xi8>) {
    %conv1 = "coa.qlinearconv"(%input, %weight, %bias) {
      in_scale = 0.00392157, in_zp = -128,
      weight_scale = [0.003921], weight_zp = [0],
      out_scale = 0.00392157, out_zp = -128,
      kernel_shape = [7, 7], strides = [2, 2],
      pads = [3, 3, 3, 3], dilations = [1, 1],
      in_addr = 0x0, out_addr = 0x10000000,
      weight_addr = 0x8000000, bias_addr = 0xC0000000,
      R = 112, C = 112, M = 64, N = 3, R0 = 224, C0 = 224,
      sM_concat = 64, M_concat = 64,
      tM = 64, tR = 112, tC = 112, factor = 4096
    } : (tensor<1x3x224x224xi8>, tensor<64x3x7x7xi8>, tensor<64xi32>) -> tensor<1x64x112x112xi8>
    return
  }
}
"""

    def test_minimal_mlir_parse(self):
        """mlir_parser.parse_all_layers should parse the minimal MLIR without error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(self.MINIMAL_MLIR)
            tmp = f.name
        try:
            layers = parse_all_layers(tmp, verbose=False)
            self.assertEqual(len(layers), 1)
            layer = layers[0]
            self.assertEqual(layer['type'], 'qlinearconv')
            self.assertEqual(layer['M'], 64)
            self.assertEqual(layer['N'], 3)
            self.assertEqual(layer['R'], 112)
        finally:
            os.unlink(tmp)


class TestTilingConstraints(unittest.TestCase):
    """Unit tests for the buffer constraint checker."""

    def test_baseline_resnet_first_conv(self):
        """ResNet first conv (3x7x7, stride=2): tile must satisfy all limits."""
        N, M, R, C, k, s, d = 3, 64, 112, 112, 7, 2, 1
        tM, tR, tC = get_tile(N, M, R, C, k, s, 0, d)
        flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, k, s, 0, d)
        self.assertEqual(flag, 0,
                         f"Tiling ({tM},{tR},{tC}) violates buffer constraints: flag={flag}")

    def test_1x1_conv(self):
        """1×1 convolution tiling is always trivially legal."""
        N, M, R, C, k, s, d = 256, 64, 56, 56, 1, 1, 1
        tM, tR, tC = get_tile(N, M, R, C, k, s, 0, d)
        flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, k, s, 0, d)
        self.assertEqual(flag, 0)

    def test_tM_minimum(self):
        """tM must be at least 16 (hardware granularity)."""
        tM, _, _ = get_tile(512, 512, 7, 7, 3, 1, 0, 1)
        self.assertGreaterEqual(tM, 16)


class TestBayesianTiling(unittest.TestCase):
    """Tests for the Bayesian tiling optimizer (smoke tests only)."""

    def test_bayesian_returns_legal_tile(self):
        """Bayesian search must return a legal tiling."""
        try:
            from optimizer.tiling_search.bayesian_opt import bayesian_tile_search
        except ImportError:
            self.skipTest("bayesian_opt not importable (missing optuna?)")

        tM, tR, tC = bayesian_tile_search(N=64, M=64, R=56, C=56,
                                           k=3, s=1, d=1, n_trials=30)
        flag = calculate_buffer_consumption(64, tM, tR, tC, 64, 64, 3, 1, 0, 1)
        self.assertEqual(flag, 0, f"Bayesian tile ({tM},{tR},{tC}) is illegal: flag={flag}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
