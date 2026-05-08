"""
quant_export_data.py — collect per-layer quantization metrics and dump to JSON.

Outputs: docs/reports/quant_data.json
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import onnx
from onnx import numpy_helper
from coa.quantize import quantize_onnx, _GraphRunner

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "reports")
REPORT_DIR = os.path.abspath(REPORT_DIR)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── metrics ──────────────────────────────────────────────────────
def snr_db(ref, test):
    r = ref.flatten().astype(np.float64)
    t = test.flatten().astype(np.float64)
    noise = r - t
    sp = np.sum(r**2)
    np_ = np.sum(noise**2)
    return 100.0 if np_ < 1e-30 else float(10 * np.log10(sp / (np_ + 1e-30)))

def cosine(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))

def error_metrics(ref, test):
    """Return dict of absolute/relative error metrics."""
    r = ref.flatten().astype(np.float64)
    t = test.flatten().astype(np.float64)
    err = r - t
    mae    = float(np.mean(np.abs(err)))
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    max_ae = float(np.max(np.abs(err)))
    # Mean relative error: element-wise |err| / (|ref| + eps)
    eps = max(float(np.mean(np.abs(r))) * 1e-3, 1e-8)
    mre    = float(np.mean(np.abs(err) / (np.abs(r) + eps)) * 100)   # %
    # Max relative error (same denominator)
    max_re = float(np.max(np.abs(err) / (np.abs(r) + eps)) * 100)    # %
    # Normalised RMSE = RMSE / std(ref)
    std_r  = float(np.std(r))
    nrmse  = float(rmse / (std_r + 1e-12) * 100)                     # %
    return dict(mae=mae, rmse=rmse, max_ae=max_ae,
                mean_re_pct=mre, max_re_pct=max_re, nrmse_pct=nrmse)

def sim_qdq(x, scale, zp, signed=True):
    lo, hi = (-128, 127) if signed else (0, 255)
    q = np.clip(np.round(x.astype(np.float64) / scale) + zp, lo, hi).astype(np.int64)
    return ((q - zp) * scale).astype(np.float32)

# ── stage / short-name helpers ────────────────────────────────────
def stage_of(tensor_name, op):
    n = tensor_name.lower()
    if "layer4" in n: return "Layer 4"
    if "layer3" in n: return "Layer 3"
    if "layer2" in n: return "Layer 2"
    if "layer1" in n: return "Layer 1"
    if op in ("GlobalAveragePool", "Flatten", "Gemm"): return "Head"
    return "Stem"

def short_name(tensor_name, op):
    parts = [p for p in tensor_name.split("/") if p and not p.startswith("_")]
    # Find last meaningful part before _output_N
    for p in reversed(parts):
        clean = p.replace("_output_0", "").replace("_output_1", "")
        if clean and clean != tensor_name:
            return clean
    return op

# ── build / quantize model ────────────────────────────────────────
def build_float_onnx():
    import torch, torchvision
    path = os.path.join(REPORT_DIR, "resnet18_float.onnx")
    if not os.path.exists(path):
        print("Exporting float ResNet-18...")
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
        dummy = torch.zeros(1, 3, 64, 64)
        torch.onnx.export(model, dummy, path,
                          input_names=["input"], output_names=["output"],
                          opset_version=13,
                          dynamic_axes={"input": {0:"b",2:"H",3:"W"},
                                        "output": {0:"b"}})
    return path

CONFIGS = [
    ("INT8-PC-MinMax",    "int8×int8\nper-ch minmax",      dict(act_format="int8",  weight_per_channel=True,  calibration="minmax",   hw_aware=False)),
    ("INT8-PC-MinMax-HW", "int8×int8\nper-ch minmax\n+HW-aware", dict(act_format="int8",  weight_per_channel=True,  calibration="minmax",   hw_aware=True)),
    ("INT8-PC-Entropy",   "int8×int8\nper-ch entropy",     dict(act_format="int8",  weight_per_channel=True,  calibration="entropy",  hw_aware=False)),
    ("UINT8-PC-MinMax",   "uint8×int8\nper-ch minmax",     dict(act_format="uint8", weight_per_channel=True,  calibration="minmax",   hw_aware=False)),
    ("INT8-PT-MinMax",    "int8×int8\nper-ten minmax",     dict(act_format="int8",  weight_per_channel=False, calibration="minmax",   hw_aware=False)),
]

LAYER_OPS = ("Conv", "Gemm", "Add", "Relu", "MaxPool", "GlobalAveragePool", "Flatten")

def collect():
    np.random.seed(42)
    float_onnx = build_float_onnx()
    calib = np.random.randn(4, 3, 64, 64).astype(np.float32)
    test_inp = np.random.randn(1, 3, 64, 64).astype(np.float32)

    float_model = onnx.load(float_onnx)
    runner = _GraphRunner(float_model)
    print("Running float inference...")
    float_env = runner.run({"input": test_inp})
    float_output_name = float_model.graph.output[0].name
    float_top1 = int(np.argmax(float_env[float_output_name].flatten()))
    init_names = {i.name for i in float_model.graph.initializer}

    # Identify layer nodes in order
    layer_nodes = [(i, n) for i, n in enumerate(float_model.graph.node)
                   if n.op_type in LAYER_OPS]

    summary_rows = []
    config_layer_data = {}

    for tag, label, kwargs in CONFIGS:
        q_path = os.path.join(REPORT_DIR, f"q_{tag}.onnx")
        print(f"Quantizing: {tag} ...", end=" ", flush=True)
        t0 = time.time()
        quantize_onnx(float_onnx, q_path, calib, verbose=False, **kwargs)
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

        q_model = onnx.load(q_path)
        q_inits = {i.name: numpy_helper.to_array(i) for i in q_model.graph.initializer}
        signed = (kwargs["act_format"] == "int8")

        def qdq(tensor_name, ft):
            s_arr = q_inits.get(tensor_name + "_scale")
            if s_arr is None or ft is None:
                return None, None
            s = float(s_arr.flat[0])
            z_arr = q_inits.get(tensor_name + "_zp")
            z = int(z_arr.flat[0]) if z_arr is not None else 0
            return snr_db(ft, sim_qdq(ft, s, z, signed)), cosine(ft, sim_qdq(ft, s, z, signed))

        # First conv / residual add / fc nodes
        conv_nodes  = [n for n in float_model.graph.node if n.op_type == "Conv"]
        gemm_nodes  = [n for n in float_model.graph.node if n.op_type == "Gemm"]
        act_add_nodes = [n for n in float_model.graph.node
                         if n.op_type == "Add"
                         and n.input[0] not in init_names
                         and n.input[1] not in init_names]

        snr_c, _ = qdq(conv_nodes[0].output[0],  float_env.get(conv_nodes[0].output[0]))
        snr_a, cos_a = qdq(act_add_nodes[0].output[0], float_env.get(act_add_nodes[0].output[0]))
        snr_g, cos_g = qdq(gemm_nodes[-1].output[0], float_env.get(gemm_nodes[-1].output[0]))
        snr_e, _    = qdq(float_output_name, float_env.get(float_output_name))

        ft_out = float_env.get(float_output_name)
        s_arr = q_inits.get(float_output_name + "_scale")
        if s_arr and ft_out is not None:
            s = float(s_arr.flat[0])
            z_arr = q_inits.get(float_output_name + "_zp")
            z = int(z_arr.flat[0]) if z_arr is not None else 0
            q_top1 = int(np.argmax(sim_qdq(ft_out, s, z, signed).flatten()))
        else:
            q_top1 = float_top1

        summary_rows.append({
            "tag": tag, "label": label,
            "snr_first_conv": round(snr_c, 2) if snr_c else None,
            "snr_first_add":  round(snr_a, 2) if snr_a else None,
            "cos_first_add":  round(cos_a, 6) if cos_a else None,
            "snr_fc":         round(snr_g, 2) if snr_g else None,
            "cos_fc":         round(cos_g, 6) if cos_g else None,
            "snr_e2e":        round(snr_e, 2) if snr_e else None,
            "top1_match":     bool(q_top1 == float_top1),
            "quant_time_s":   round(elapsed, 2),
        })

        # Per-layer metrics
        layers = []
        for global_idx, (node_idx, node) in enumerate(layer_nodes):
            out_name = node.output[0]
            ft = float_env.get(out_name)
            s_val, cos_val = qdq(out_name, ft) if ft is not None else (None, None)

            # Absolute / relative error
            errs = {}
            if ft is not None and s_val is not None:
                s_arr = q_inits.get(out_name + "_scale")
                if s_arr is not None:
                    s = float(s_arr.flat[0])
                    z_arr = q_inits.get(out_name + "_zp")
                    z = int(z_arr.flat[0]) if z_arr is not None else 0
                    sim = sim_qdq(ft, s, z, signed)
                    raw = error_metrics(ft, sim)
                    errs = {k: round(v, 6) for k, v in raw.items()}

            layers.append({
                "global_idx": global_idx,
                "node_idx":   node_idx,
                "op":         node.op_type,
                "short_name": short_name(out_name, node.op_type),
                "tensor":     out_name,
                "stage":      stage_of(out_name, node.op_type),
                "snr":        round(s_val, 2) if s_val is not None else None,
                "cosine":     round(cos_val, 6) if cos_val is not None else None,
                **errs,
            })
        config_layer_data[tag] = layers

    data = {
        "meta": {
            "model": "ResNet-18 (torchvision pretrained)",
            "input_shape": [1, 3, 64, 64],
            "calib_n": 4,
            "float_top1": int(float_top1),
        },
        "configs": [{"tag": r["tag"], "label": r["label"]} for r in summary_rows],
        "summary": summary_rows,
        "per_layer": config_layer_data,
    }

    out_path = os.path.join(REPORT_DIR, "quant_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nData saved → {out_path}")
    return out_path

if __name__ == "__main__":
    collect()
