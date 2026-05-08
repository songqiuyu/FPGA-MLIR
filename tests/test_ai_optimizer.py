"""
Unit tests for the AI optimizer modules (Phase 3-A/B/C).
All tests use lightweight synthetic data and do not require GPU or trained models.
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np

AI_DIR = os.path.join(os.path.dirname(__file__), '..', 'optimizer')


class TestTilingEnv(unittest.TestCase):
    """Tests for the RL tiling environment."""

    def setUp(self):
        sys.path.insert(0, os.path.join(AI_DIR, 'tiling_search'))

    def test_reset_returns_correct_shape(self):
        from env import TilingEnv
        env = TilingEnv(N=64, M=64, R=56, C=56, k=3, s=1, d=1)
        obs = env.reset()
        self.assertEqual(obs.shape, (TilingEnv.STATE_DIM,))

    def test_all_actions_executable(self):
        from env import TilingEnv
        env = TilingEnv(N=64, M=64, R=56, C=56, k=3, s=1, d=1)
        for action in range(TilingEnv.N_ACTIONS):
            env.reset()
            obs, reward, done, _ = env.step(action)
            self.assertEqual(obs.shape, (TilingEnv.STATE_DIM,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)

    def test_done_action_terminates(self):
        from env import TilingEnv
        env = TilingEnv(N=64, M=64, R=56, C=56)
        env.reset()
        _, _, done, _ = env.step(8)  # action 8 = done
        self.assertTrue(done)

    def test_initial_tile_equals_full_dims(self):
        from env import TilingEnv
        env = TilingEnv(N=64, M=128, R=28, C=28)
        env.reset()
        self.assertEqual(env.tM, 128)
        self.assertEqual(env.tR, 28)
        self.assertEqual(env.tC, 28)

    def test_buffer_utilization_legal(self):
        """Buffer utilization must be <= 1.0 after get_tile() produces a legal tile."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legacy', 'tools'))
        from assign_addr import get_tile
        from env import buffer_utilization
        N, M, R, C, k, s, d = 64, 128, 28, 28, 3, 1, 1
        tM, tR, tC = get_tile(N, M, R, C, k, s, 0, d)
        wU, gU, oU = buffer_utilization(N, tM, tR, tC, N, M, k, s, d)
        self.assertLessEqual(wU, 1.0, f"wdepth over limit with tile ({tM},{tR},{tC})")
        self.assertLessEqual(gU, 1.0, f"gdepth over limit with tile ({tM},{tR},{tC})")
        self.assertLessEqual(oU, 1.0, f"odepth over limit with tile ({tM},{tR},{tC})")


class TestGraphBuilder(unittest.TestCase):
    """Tests for op-fusion graph construction."""

    SAMPLE_MLIR = """\
module {
  func.func @main(%x: tensor<1x3x224x224xi8>) {
    %conv1 = "coa.qlinearconv"(%x, %w1, %b1) {
      in_scale = 0.004, in_zp = -128, weight_scale = [0.003], weight_zp = [0],
      out_scale = 0.004, out_zp = -128, kernel_shape = [3, 3],
      strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1],
      R = 56, C = 56, M = 64, N = 64, R0 = 56, C0 = 56,
      tM = 64, tR = 56, tC = 56, factor = 4096,
      in_addr = 0x10000000, out_addr = 0x10000000,
      weight_addr = 0x8000000, bias_addr = 0xC0000000
    } : () -> tensor<1x64x56x56xi8>
    %add1 = "coa.qlinearadd"(%conv1, %conv1) {
      a_scale = 0.004, a_zp = -128, b_scale = 0.004, b_zp = -128,
      out_scale = 0.004, out_zp = -128,
      R = 56, C = 56, M = 64, N = 64, R0 = 56, C0 = 56,
      tM = 64, tR = 56, tC = 56, factor = 32768, factor2 = 32768,
      in_addr = 0x10000000, in2_addr = 0x10000000, out_addr = 0x10000000
    } : () -> tensor<1x64x56x56xi8>
    return
  }
}
"""

    def setUp(self):
        sys.path.insert(0, os.path.join(AI_DIR, 'op_fusion'))

    def _write_mlir(self, text: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir',
                                         delete=False, encoding='utf-8') as f:
            f.write(text)
            return f.name

    def test_graph_parsing(self):
        from graph_builder import build_graph_from_mlir
        tmp = self._write_mlir(self.SAMPLE_MLIR)
        try:
            g = build_graph_from_mlir(tmp)
            self.assertIsNotNone(g)
            self.assertEqual(len(g["op_names"]), 2)
            self.assertEqual(g["node_features"].shape[1], 15)
        finally:
            os.unlink(tmp)

    def test_edge_construction(self):
        from graph_builder import build_graph_from_mlir
        tmp = self._write_mlir(self.SAMPLE_MLIR)
        try:
            g = build_graph_from_mlir(tmp)
            # conv1 -> add1 should be detected as an edge
            self.assertGreater(g["edge_index"].shape[1], 0)
        finally:
            os.unlink(tmp)

    def test_candidate_pairs(self):
        from graph_builder import build_graph_from_mlir, enumerate_candidate_pairs
        tmp = self._write_mlir(self.SAMPLE_MLIR)
        try:
            g = build_graph_from_mlir(tmp)
            pairs = enumerate_candidate_pairs(g)
            self.assertIsInstance(pairs, list)
            # Each pair should be (int, int, str)
            for src, dst, label in pairs:
                self.assertIsInstance(src, int)
                self.assertIsInstance(dst, int)
                self.assertIn("->", label)
        finally:
            os.unlink(tmp)


class TestSensitivity(unittest.TestCase):
    """Tests for AutoQuant sensitivity analysis."""

    def setUp(self):
        sys.path.insert(0, os.path.join(AI_DIR, 'auto_quant'))

    def _make_dummy_npz(self) -> str:
        """Create a temporary .npz file with synthetic weights."""
        data = {
            "layer1.weight": np.random.randn(64, 64, 3, 3).astype(np.float32),
            "layer2.weight": np.random.randn(128, 64, 1, 1).astype(np.float32),
            "layer3.bias":   np.random.randn(64).astype(np.float32),   # 1-D -> skip
        }
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f, **data)
            return f.name

    def test_sensitivity_computation(self):
        from sensitivity import compute_layer_sensitivity
        npz = self._make_dummy_npz()
        try:
            results = compute_layer_sensitivity(npz)
            # Only 2-D+ tensors: layer1.weight, layer2.weight (not bias)
            self.assertEqual(len(results), 2)
            for r in results:
                self.assertIn(4,  r["sensitivity"])
                self.assertIn(8,  r["sensitivity"])
                self.assertIn(16, r["sensitivity"])
                # 4-bit error > 8-bit error > 16-bit error (>= for numerical safety)
                self.assertGreaterEqual(r["sensitivity"][4], r["sensitivity"][8])
                self.assertGreaterEqual(r["sensitivity"][8], r["sensitivity"][16])
        finally:
            os.unlink(npz)

    def test_sensitivity_ranking(self):
        from sensitivity import compute_layer_sensitivity, rank_layers_by_sensitivity
        npz = self._make_dummy_npz()
        try:
            sens    = compute_layer_sensitivity(npz)
            ranked  = rank_layers_by_sensitivity(sens, bits_ref=4)
            self.assertEqual(len(ranked), len(sens))
            # Ranks should be sequential starting from 0
            for i, r in enumerate(ranked):
                self.assertEqual(r["rank"], i)
            # First element has highest 4-bit error
            if len(ranked) > 1:
                self.assertGreaterEqual(ranked[0]["sensitivity"][4],
                                        ranked[1]["sensitivity"][4])
        finally:
            os.unlink(npz)

    def test_bit_assignment_budget(self):
        from sensitivity import (compute_layer_sensitivity,
                                 rank_layers_by_sensitivity, assign_bits_budget)
        npz = self._make_dummy_npz()
        try:
            sens   = compute_layer_sensitivity(npz)
            ranked = rank_layers_by_sensitivity(sens)
            assignment = assign_bits_budget(ranked, bit_budget=6.0)
            self.assertEqual(len(assignment), len(sens))
            for v in assignment.values():
                self.assertIn(v, [4, 8, 16])
        finally:
            os.unlink(npz)


class TestMixedQuant(unittest.TestCase):
    """Tests for mixed-precision MLIR rewriting."""

    MLIR = ('    %conv1 = "coa.qlinearconv"(%input, %weight_layer1, %bias)'
             ' { in_scale = 0.00392157, in_zp = -128,'
             ' weight_scale = [0.003], weight_zp = [0],'
             ' out_scale = 0.00392157, out_zp = -128,'
             ' kernel_shape = [3, 3], strides = [1, 1],'
             ' pads = [1, 1, 1, 1], dilations = [1, 1],'
             ' R = 56, C = 56, M = 64, N = 64, R0 = 56, C0 = 56,'
             ' tM = 64, tR = 56, tC = 56, factor = 4096,'
             ' in_addr = 0x10000000, out_addr = 0x10000000,'
             ' weight_addr = 0x8000000, bias_addr = 0xC0000000'
             ' } : ()')

    def setUp(self):
        sys.path.insert(0, os.path.join(AI_DIR, 'auto_quant'))

    def test_8bit_unchanged(self):
        from mixed_quant import apply_mixed_precision
        bit_assignment = {"weight_layer1": 8}
        out = apply_mixed_precision(self.MLIR, bit_assignment)
        # 8-bit: line should be unchanged (no comment added)
        self.assertNotIn("mixed-prec", out)

    def test_4bit_modifies_scale(self):
        from mixed_quant import apply_mixed_precision
        bit_assignment = {"weight_layer1": 4}
        out = apply_mixed_precision(self.MLIR, bit_assignment)
        self.assertIn("mixed-prec: 4-bit", out)
        # Scale should be numerically different
        import re
        orig_scale = 0.00392157
        m = re.search(r'in_scale\s*=\s*([\d.eE+\-]+)', out)
        if m:
            new_scale = float(m.group(1))
            self.assertNotAlmostEqual(new_scale, orig_scale, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
