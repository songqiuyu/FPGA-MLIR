"""
compare_hw_params.py — Compare hardware parameters from two quantization pipelines.

Pipeline A (ours):   coa.quantize_onnx  (INT8-PT-MinMax) + coa.hw_export
Pipeline B (legacy): onnxruntime.quantize_static (per-tensor, MinMax) + coa.hw_export

Both pipelines use:
  model : examples/resnet18/model/resnet18.onnx
  calib : legacy/datasets/calibration_data/*.npy

Output:
  examples/resnet18/parameters/ours/    — pipeline A hw params
  examples/resnet18/parameters/legacy_hw/ — pipeline B hw params
  Comparison report printed to stdout.
"""

import os, sys, tempfile
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODEL_PATH  = os.path.join(ROOT, "examples", "resnet18", "model", "resnet18.onnx")
CALIB_DIR   = os.path.join(ROOT, "legacy", "datasets", "calibration_data")
OUT_OURS    = os.path.join(ROOT, "examples", "resnet18", "parameters", "ours")
OUT_LEGACY  = os.path.join(ROOT, "examples", "resnet18", "parameters", "legacy_hw")
LEGACY_REF  = os.path.join(ROOT, "legacy", "parameters")

# ── 1. Load calibration data ──────────────────────────────────────────────────
def load_calib(calib_dir: str) -> np.ndarray:
    files = sorted(f for f in os.listdir(calib_dir) if f.endswith(".npy"))
    batches = [np.load(os.path.join(calib_dir, f)) for f in files]
    data = np.concatenate(batches, axis=0)
    print(f"[calib] loaded {len(batches)} files → {data.shape}  dtype={data.dtype}")
    return data


# ── 2. Pipeline A: coa.quantize_onnx ─────────────────────────────────────────
def run_pipeline_a(calib: np.ndarray) -> str:
    from coa.quantize import quantize_onnx
    out_onnx = os.path.join(ROOT, "examples", "resnet18", "parameters",
                             "q_INT8-PT-MinMax-legacy-calib.onnx")
    print("\n[Pipeline A] coa.quantize_onnx INT8-PT-MinMax …")
    quantize_onnx(
        MODEL_PATH, out_onnx, calib,
        act_format="int8",
        weight_per_channel=False,
        calibration="minmax",
        verbose=False,
    )
    print(f"  → {out_onnx}")
    return out_onnx


# ── 3. Pipeline B: onnxruntime quantize_static ────────────────────────────────
def run_pipeline_b() -> str:
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
    from onnxruntime.quantization import CalibrationDataReader

    class _NpyReader(CalibrationDataReader):
        def __init__(self, calib_dir, model_path):
            import onnx as _onnx
            self._dir   = calib_dir
            self._files = sorted(f for f in os.listdir(calib_dir) if f.endswith(".npy"))
            self._iter  = iter(self._files)
            m = _onnx.load(model_path)
            self._input_name = m.graph.input[0].name
        def get_next(self):
            try:
                fname = next(self._iter)
            except StopIteration:
                return None
            data = np.load(os.path.join(self._dir, fname))
            return {self._input_name: data}
        def rewind(self):
            self._iter = iter(self._files)

    out_onnx = os.path.join(ROOT, "examples", "resnet18", "parameters",
                             "q_legacy_quantize_static.onnx")
    print("\n[Pipeline B] onnxruntime quantize_static per-tensor MinMax …")
    quantize_static(
        model_input=MODEL_PATH,
        model_output=out_onnx,
        calibration_data_reader=_NpyReader(CALIB_DIR, MODEL_PATH),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"  → {out_onnx}")
    return out_onnx


# ── 4. hw_export for both ─────────────────────────────────────────────────────
def run_hw_export(onnx_path: str, out_dir: str):
    from coa.hw_export import export_hw_params
    print(f"\n[hw_export] {os.path.basename(onnx_path)} → {out_dir}")
    export_hw_params(onnx_path, out_dir, force_per_tensor=True, verbose=False)


# ── 5. Compare two directories ────────────────────────────────────────────────
def compare_dirs(dir_a: str, dir_b: str, label_a: str, label_b: str):
    files_to_compare = ["weight.image", "bias.image", "act.image"]
    print(f"\n{'='*70}")
    print(f"COMPARISON: {label_a}  vs  {label_b}")
    print(f"{'='*70}")

    for fname in files_to_compare:
        path_a = os.path.join(dir_a, fname)
        path_b = os.path.join(dir_b, fname)
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"\n  {fname}: MISSING (a={os.path.exists(path_a)}, b={os.path.exists(path_b)})")
            continue

        ba = np.frombuffer(open(path_a, "rb").read(), dtype=np.uint8)
        bb = np.frombuffer(open(path_b, "rb").read(), dtype=np.uint8)
        print(f"\n  {fname}:")
        print(f"    size  A={len(ba):,}  B={len(bb):,}", end="")
        if len(ba) != len(bb):
            print(f"  ← SIZE MISMATCH")
            continue
        print()

        if len(ba) == 0:
            print("    (empty — no data written)  ✓")
            continue
        diff = (ba != bb)
        n_diff = int(diff.sum())
        pct = 100.0 * n_diff / len(ba)
        print(f"    diff  {n_diff}/{len(ba)} bytes ({pct:.2f}%)", end="")
        if n_diff == 0:
            print("  ✓ IDENTICAL")
        else:
            print(f"  ✗ DIFFER")
            # Show first few differing positions
            idxs = np.where(diff)[0][:5]
            for i in idxs:
                print(f"    [byte {i:08x}]  A=0x{int(ba[i]):02x}  B=0x{int(bb[i]):02x}")


# ── 6. Compare with legacy/parameters/ (pre-built reference) ─────────────────
def compare_with_legacy_ref(our_dir: str):
    print(f"\n{'='*70}")
    print(f"COMPARISON: ours  vs  legacy/parameters/ (pre-built reference)")
    print(f"{'='*70}")

    # weight.image
    for fname_ours, fname_ref, note in [
        ("weight.image", "weight.image", "INT8 weights (OC,KH,KW,IC_pad32)"),
        ("bias.image",   "bias.image",   "INT32 biases (OC_pad16)"),
        ("act.image",    "relu.image",   "activation LUT (ours=real LUT, legacy=identity)"),
    ]:
        pa = os.path.join(our_dir, fname_ours)
        pb = os.path.join(LEGACY_REF, fname_ref)
        if not os.path.exists(pa) or not os.path.exists(pb):
            print(f"\n  {fname_ours}/{fname_ref}: MISSING")
            continue

        ba = np.frombuffer(open(pa, "rb").read(), dtype=np.uint8)
        bb = np.frombuffer(open(pb, "rb").read(), dtype=np.uint8)

        print(f"\n  {fname_ours} vs {fname_ref}  [{note}]:")
        print(f"    size  ours={len(ba):,}  legacy={len(bb):,}", end="")
        if len(ba) != len(bb):
            print(f"  ← SIZE MISMATCH (expected, file covers different #layers)")
        else:
            diff = (ba != bb)
            n_diff = int(diff.sum())
            pct = 100.0 * n_diff / len(ba)
            print(f"\n    diff  {n_diff}/{len(ba)} bytes ({pct:.2f}%)", end="")
            if n_diff == 0:
                print("  ✓ IDENTICAL")
            else:
                print(f"  ✗ DIFFER")
                if "weight" in fname_ours:
                    # Show per-layer diff statistics for weights
                    # weight.image is packed (OC,KH,KW,IC_pad32); we can't easily split
                    # just show byte diff histogram
                    vals_a = ba.astype(np.int16) - 128   # back to int8 range approx
                    vals_b = bb.astype(np.int16) - 128
                    delta = np.abs(vals_a - vals_b)
                    print(f"\n    |Δ| stats: mean={delta.mean():.3f}  max={delta.max()}"
                          f"  nonzero={int((delta>0).sum())}")
                idxs = np.where(diff)[0][:5]
                for i in idxs:
                    print(f"\n    [byte {i:08x}]  ours=0x{int(ba[i]):02x}  legacy=0x{int(bb[i]):02x}")


# ── 7. Print scale comparison ─────────────────────────────────────────────────
def compare_scales(onnx_a_path: str, onnx_b_path: str, label_a: str, label_b: str):
    import onnx
    from onnx import numpy_helper

    def get_scales(path):
        m = onnx.load(path)
        inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
        scales = {}
        for name, arr in inits.items():
            if name.endswith("_scale") and arr.size == 1:
                # activation scale
                base = name[:-len("_scale")]
                scales[base] = float(arr.flat[0])
        return scales

    sa = get_scales(onnx_a_path)
    sb = get_scales(onnx_b_path)

    print(f"\n{'='*70}")
    print(f"ACTIVATION SCALE COMPARISON: {label_a} vs {label_b}")
    print(f"  (per-tensor weight scales excluded; only scalar activation scales shown)")
    print(f"{'='*70}")
    common = sorted(set(sa) & set(sb))
    n_match = 0
    print(f"  {'Tensor':<55} {'A':>12}  {'B':>12}  {'rel%':>8}")
    print(f"  {'-'*55} {'-'*12}  {'-'*12}  {'-'*8}")
    for k in common:
        va, vb = sa[k], sb[k]
        rel = abs(va-vb)/(abs(va)+1e-12)*100
        mark = "  ✓" if rel < 1.0 else f"  ← {rel:.1f}%"
        print(f"  {k:<55} {va:>12.6f}  {vb:>12.6f}  {rel:>7.2f}%{mark}")
        if rel < 5.0:
            n_match += 1
    print(f"\n  {n_match}/{len(common)} tensors match within 5%")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quant", action="store_true",
                    help="skip quantization (use existing ONNX files)")
    ap.add_argument("--skip-export", action="store_true",
                    help="skip hw_export (use existing parameter files)")
    args = ap.parse_args()

    os.makedirs(OUT_OURS,   exist_ok=True)
    os.makedirs(OUT_LEGACY, exist_ok=True)

    onnx_a = os.path.join(ROOT, "examples", "resnet18", "parameters",
                           "q_INT8-PT-MinMax-legacy-calib.onnx")
    onnx_b = os.path.join(ROOT, "examples", "resnet18", "parameters",
                           "q_legacy_quantize_static.onnx")

    if not args.skip_quant:
        calib = load_calib(CALIB_DIR)
        onnx_a = run_pipeline_a(calib)
        onnx_b = run_pipeline_b()

    if not args.skip_export:
        run_hw_export(onnx_a, OUT_OURS)
        run_hw_export(onnx_b, OUT_LEGACY)

    # Compare A vs B (both go through hw_export)
    compare_dirs(OUT_OURS, OUT_LEGACY, "coa.quantize (ours)", "onnxruntime.quantize_static (legacy)")

    # Compare ours vs pre-built legacy reference
    compare_with_legacy_ref(OUT_OURS)

    # Scale comparison
    if os.path.exists(onnx_a) and os.path.exists(onnx_b):
        compare_scales(onnx_a, onnx_b,
                       "coa.quantize (ours)",
                       "onnxruntime.quantize_static")
