"""
Tests for coa.quantize — self-contained PTQ quantization toolkit.
"""

import sys
import os
import unittest

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coa.quantize import (
    compute_scale_zp_symmetric,
    compute_scale_zp_asymmetric,
    quantize_array,
    dequantize_array,
    smooth_quant_scales,
    _calib_minmax,
    _calib_percentile,
    _calib_entropy,
    _quantize_weight_per_channel,
    _quantize_weight_per_tensor,
    ActFormat,
    CalibMethod,
    WeightScheme,
)


class TestScaleZeroPoint(unittest.TestCase):
    """Test scale/zero-point computation for symmetric and asymmetric modes."""

    def test_symmetric_basic(self):
        s, z = compute_scale_zp_symmetric(-1.0, 1.0, bits=8)
        self.assertEqual(z, 0, "Symmetric quant should have zp=0")
        self.assertAlmostEqual(s, 1.0 / 127, places=6)

    def test_symmetric_positive_only(self):
        s, z = compute_scale_zp_symmetric(0.0, 6.0, bits=8)
        self.assertEqual(z, 0)
        self.assertAlmostEqual(s, 6.0 / 127, places=6)

    def test_symmetric_negative_only(self):
        s, z = compute_scale_zp_symmetric(-3.0, 0.0, bits=8)
        self.assertEqual(z, 0)
        self.assertAlmostEqual(s, 3.0 / 127, places=6)

    def test_asymmetric_basic(self):
        s, z = compute_scale_zp_asymmetric(0.0, 1.0, bits=8)
        self.assertGreater(s, 0)
        self.assertGreaterEqual(z, 0)
        self.assertLessEqual(z, 255)

    def test_asymmetric_negative_range(self):
        s, z = compute_scale_zp_asymmetric(-2.0, 2.0, bits=8)
        self.assertGreater(s, 0)
        # z should be near 128 for a centered range
        self.assertTrue(100 < z < 155, f"zp={z} should be near 128")

    def test_asymmetric_all_positive(self):
        s, z = compute_scale_zp_asymmetric(0.0, 6.0, bits=8)
        self.assertEqual(z, 0, "All-positive range should have zp=0")

    def test_4bit_symmetric(self):
        s, z = compute_scale_zp_symmetric(-1.0, 1.0, bits=4)
        self.assertEqual(z, 0)
        self.assertAlmostEqual(s, 1.0 / 7, places=5)


class TestQuantizeDequantize(unittest.TestCase):
    """Test quantize and dequantize round-trip."""

    def test_roundtrip_symmetric(self):
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        s, z = compute_scale_zp_symmetric(-1.0, 1.0, bits=8)
        q = quantize_array(x, s, z, signed=True, bits=8)
        x_hat = dequantize_array(q, s, z)
        np.testing.assert_allclose(x, x_hat, atol=s + 1e-6)

    def test_roundtrip_asymmetric(self):
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        s, z = compute_scale_zp_asymmetric(0.0, 1.0, bits=8)
        q = quantize_array(x, s, z, signed=False, bits=8)
        x_hat = dequantize_array(q, s, z)
        np.testing.assert_allclose(x, x_hat, atol=s + 1e-6)

    def test_clipping(self):
        x = np.array([-200.0, 200.0], dtype=np.float32)
        s, z = compute_scale_zp_symmetric(-1.0, 1.0, bits=8)
        q = quantize_array(x, s, z, signed=True, bits=8)
        self.assertEqual(q[0], -128)
        self.assertEqual(q[1], 127)

    def test_unsigned_clipping(self):
        x = np.array([-1.0, 2.0], dtype=np.float32)
        s, z = compute_scale_zp_asymmetric(0.0, 1.0, bits=8)
        q = quantize_array(x, s, z, signed=False, bits=8)
        self.assertEqual(q[0], 0)
        self.assertEqual(q[1], 255)


class TestWeightQuantization(unittest.TestCase):
    """Test per-channel and per-tensor weight quantization."""

    def test_per_channel_shape(self):
        W = np.random.randn(64, 3, 3, 3).astype(np.float32)
        W_q, scales, zps = _quantize_weight_per_channel(W, axis=0)
        self.assertEqual(W_q.shape, W.shape)
        self.assertEqual(len(scales), 64)
        self.assertEqual(len(zps), 64)
        self.assertTrue(np.all(zps == 0), "Symmetric quant → all zps should be 0")

    def test_per_channel_range(self):
        W = np.random.randn(32, 16, 1, 1).astype(np.float32) * 5
        W_q, scales, zps = _quantize_weight_per_channel(W)
        self.assertTrue(np.all(W_q >= -128))
        self.assertTrue(np.all(W_q <= 127))

    def test_per_tensor_range(self):
        W = np.random.randn(64, 64, 3, 3).astype(np.float32)
        W_q, s, z = _quantize_weight_per_tensor(W)
        self.assertEqual(W_q.shape, W.shape)
        self.assertTrue(np.all(W_q >= -128))
        self.assertTrue(np.all(W_q <= 127))
        self.assertEqual(z, 0)

    def test_per_channel_reconstruction_error(self):
        W = np.random.randn(16, 8, 3, 3).astype(np.float32)
        W_q, scales, zps = _quantize_weight_per_channel(W, axis=0)
        # Dequantize per channel
        W_deq = np.zeros_like(W, dtype=np.float32)
        for c in range(W.shape[0]):
            W_deq[c] = W_q[c].astype(np.float32) * scales[c]
        max_err = np.max(np.abs(W - W_deq))
        self.assertLess(max_err, np.max(np.abs(W)) * 0.1,
                        "Per-channel reconstruction error should be < 10%")


class TestSmoothQuant(unittest.TestCase):
    """Test SmoothQuant scale computation."""

    def test_basic(self):
        act_max = np.array([10.0, 1.0, 100.0], dtype=np.float32)
        w_max   = np.array([1.0, 10.0, 1.0],   dtype=np.float32)
        s = smooth_quant_scales(act_max, w_max, alpha=0.5)
        self.assertEqual(s.shape, (3,))
        # s = act^0.5 / w^0.5 = sqrt(act/w)
        expected = np.sqrt(act_max / w_max)
        np.testing.assert_allclose(s, expected, rtol=1e-5)

    def test_alpha_zero(self):
        act_max = np.array([10.0, 1.0], dtype=np.float32)
        w_max   = np.array([1.0, 10.0], dtype=np.float32)
        s = smooth_quant_scales(act_max, w_max, alpha=0.0)
        # s = act^0 / w^1 = 1/w
        np.testing.assert_allclose(s, 1.0 / w_max, rtol=1e-5)

    def test_alpha_one(self):
        act_max = np.array([10.0, 1.0], dtype=np.float32)
        w_max   = np.array([1.0, 10.0], dtype=np.float32)
        s = smooth_quant_scales(act_max, w_max, alpha=1.0)
        # s = act^1 / w^0 = act
        np.testing.assert_allclose(s, act_max, rtol=1e-5)


class TestCalibration(unittest.TestCase):
    """Test calibration algorithms on synthetic histograms."""

    def _make_hist(self, data, n_bins=2048):
        vmin, vmax = float(data.min()), float(data.max())
        edges = np.linspace(vmin, vmax, n_bins + 1)
        hist, _ = np.histogram(data, bins=edges)
        return hist, edges

    def test_minmax(self):
        data = np.random.randn(10000).astype(np.float32)
        hist, edges = self._make_hist(data)
        lo, hi = _calib_minmax(hist, edges)
        self.assertAlmostEqual(lo, float(edges[0]), places=4)
        self.assertAlmostEqual(hi, float(edges[-1]), places=4)

    def test_percentile_narrows_range(self):
        data = np.concatenate([
            np.random.randn(9990).astype(np.float32),
            np.array([100.0, -100.0], dtype=np.float32),  # outliers
        ])
        hist, edges = self._make_hist(data)
        lo_mm, hi_mm = _calib_minmax(hist, edges)
        lo_pc, hi_pc = _calib_percentile(hist, edges, percentile=99.0)
        self.assertGreater(lo_pc, lo_mm, "Percentile should clip lower bound")
        self.assertLess(hi_pc, hi_mm, "Percentile should clip upper bound")

    def test_entropy_returns_valid(self):
        data = np.random.randn(10000).astype(np.float32)
        hist, edges = self._make_hist(data)
        lo, hi = _calib_entropy(hist, edges, num_quantized_bins=128)
        self.assertLess(lo, hi)
        self.assertGreaterEqual(lo, float(edges[0]))
        self.assertLessEqual(hi, float(edges[-1]))


class TestActFormats(unittest.TestCase):
    """Test that both int8 and uint8 activation formats produce valid results."""

    def test_int8_format(self):
        x = np.random.randn(100).astype(np.float32)
        s, z = compute_scale_zp_symmetric(float(x.min()), float(x.max()))
        q = quantize_array(x, s, z, signed=True)
        self.assertEqual(q.dtype, np.int8)
        self.assertTrue(np.all(q >= -128))
        self.assertTrue(np.all(q <= 127))

    def test_uint8_format(self):
        x = np.random.randn(100).astype(np.float32)
        s, z = compute_scale_zp_asymmetric(float(x.min()), float(x.max()))
        q = quantize_array(x, s, z, signed=False)
        self.assertEqual(q.dtype, np.uint8)
        self.assertTrue(np.all(q >= 0))
        self.assertTrue(np.all(q <= 255))


if __name__ == "__main__":
    unittest.main()
