"""
quant_report.py — quantization quality diagnostic for any float ONNX model.

Usage:
    python tools/quant_report.py                   # uses torchvision ResNet-18
    python tools/quant_report.py --model path.onnx --input-shape 1 3 224 224
"""
import sys, os, argparse, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coa.quantize import (
    quantize_onnx, _GraphRunner, CalibrationCollector, CalibMethod,
    compute_scale_zp_symmetric, compute_scale_zp_asymmetric,
    quantize_array, dequantize_array,
)
import onnx
from onnx import numpy_helper


# ── metrics ──────────────────────────────────────────────────────
def snr_db(ref, test):
    r = ref.flatten().astype(np.float64)
    t = test.flatten().astype(np.float64)
    noise = r - t
    sp = np.sum(r ** 2)
    np_ = np.sum(noise ** 2)
    return 100.0 if np_ < 1e-30 else float(10 * np.log10(sp / (np_ + 1e-30)))

def cosine(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))

def mse(a, b):
    return float(np.mean((a.flatten().astype(np.float64) - b.flatten().astype(np.float64)) ** 2))

def sim_qdq(x, scale, zp, signed=True):
    lo, hi = (-128, 127) if signed else (0, 255)
    q = np.clip(np.round(x.astype(np.float64) / scale) + zp, lo, hi).astype(np.int64)
    return ((q - zp) * scale).astype(np.float32)


# ── build / load model ────────────────────────────────────────────
def get_resnet18_onnx(tmp_dir, input_shape):
    import torch, torchvision
    float_onnx = os.path.join(tmp_dir, "resnet18_float.onnx")
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()
    dummy = torch.zeros(*input_shape)
    torch.onnx.export(model, dummy, float_onnx,
                      input_names=["input"], output_names=["output"],
                      opset_version=13,
                      dynamic_axes={"input": {0:"b",2:"H",3:"W"}, "output":{0:"b"}})
    return float_onnx


def run_report(float_onnx, input_shape, calib_n=4, out_dir="/tmp/coa_report"):
    os.makedirs(out_dir, exist_ok=True)
    calib = np.random.randn(calib_n, *input_shape[1:]).astype(np.float32)
    test_inp = np.random.randn(*input_shape).astype(np.float32)

    configs = [
        ("int8×int8  per-ch  minmax",   dict(act_format="int8",  weight_per_channel=True,  calibration="minmax")),
        ("int8×int8  per-ch  entropy",  dict(act_format="int8",  weight_per_channel=True,  calibration="entropy")),
        ("uint8×int8 per-ch  minmax",   dict(act_format="uint8", weight_per_channel=True,  calibration="minmax")),
        ("int8×int8  per-ten minmax",   dict(act_format="int8",  weight_per_channel=False, calibration="minmax")),
    ]

    float_model = onnx.load(float_onnx)
    runner = _GraphRunner(float_model)
    t0 = time.time()
    float_env = runner.run({"input": test_inp})
    print(f"Float inference:  {time.time()-t0:.2f}s")

    float_output_name = float_model.graph.output[0].name
    float_out = float_env.get(float_output_name)
    float_top1 = int(np.argmax(float_out.flatten())) if float_out is not None else -1

    init_names = {i.name for i in float_model.graph.initializer}
    conv_nodes  = [n for n in float_model.graph.node if n.op_type == "Conv"]
    gemm_nodes  = [n for n in float_model.graph.node if n.op_type == "Gemm"]
    act_add_nodes = [n for n in float_model.graph.node
                     if n.op_type == "Add"
                     and n.input[0] not in init_names
                     and n.input[1] not in init_names]

    first_conv_out = conv_nodes[0].output[0]  if conv_nodes  else None
    last_gemm_out  = gemm_nodes[-1].output[0] if gemm_nodes  else None
    first_add_out  = act_add_nodes[0].output[0] if act_add_nodes else None

    float_first_conv = float_env.get(first_conv_out)
    float_last_gemm  = float_env.get(last_gemm_out)
    float_first_add  = float_env.get(first_add_out)

    header = f"\n{'Config':<35} {'1st-Conv SNR':>13} {'ResBlk cos':>11} {'FC cos':>8} {'E2E top-1':>10} {'E2E SNR':>8} {'time':>6}"
    print(header)
    print("─" * len(header))

    for label, kwargs in configs:
        tag = label.replace(" ", "_").replace("×","x").replace("/","-")
        q_onnx = os.path.join(out_dir, f"q_{tag}.onnx")
        t0 = time.time()
        quantize_onnx(float_onnx, q_onnx, calib, verbose=False, **kwargs)
        quant_time = time.time() - t0

        q_model = onnx.load(q_onnx)
        q_inits = {i.name: numpy_helper.to_array(i) for i in q_model.graph.initializer}

        def get_sq(tensor_name, signed=True):
            s_arr = q_inits.get(tensor_name + "_scale")
            z_arr = q_inits.get(tensor_name + "_zp")
            if s_arr is None:
                return None
            s = float(s_arr.flat[0])
            z = int(z_arr.flat[0]) if z_arr is not None else 0
            return s, z, signed

        # per-layer metrics
        def layer_snr(float_tensor, name, signed=True):
            if float_tensor is None: return float("nan"), float("nan")
            sq = get_sq(name, signed)
            if sq is None: return float("nan"), float("nan")
            s, z, sig = sq
            sim = sim_qdq(float_tensor, s, z, sig)
            return snr_db(float_tensor, sim), cosine(float_tensor, sim)

        signed_act = (kwargs["act_format"] == "int8")
        snr_conv,  cos_conv  = layer_snr(float_first_conv, first_conv_out,  signed_act)
        snr_add,   cos_add   = layer_snr(float_first_add,  first_add_out,   signed_act)
        snr_gemm,  cos_gemm  = layer_snr(float_last_gemm,  last_gemm_out,   signed_act)
        snr_e2e,   cos_e2e   = layer_snr(float_out,        float_output_name, signed_act)

        sq_out = get_sq(float_output_name, signed_act)
        if sq_out:
            q_out_sim = sim_qdq(float_out, sq_out[0], sq_out[1], signed_act)
            q_top1 = int(np.argmax(q_out_sim.flatten()))
        else:
            q_top1 = -1

        top1_match = "✓" if q_top1 == float_top1 else f"✗({q_top1})"

        print(f"{label:<35} {snr_conv:>12.1f}dB {cos_add:>11.4f} {cos_gemm:>8.4f} {top1_match:>10} {snr_e2e:>7.1f}dB {quant_time:>5.1f}s")

    # ── layer-by-layer breakdown (int8 per-ch) ───────────────────
    print("\n\n=== Per-layer SNR breakdown (int8×int8 per-channel minmax) ===")
    first_tag = configs[0][0].replace(" ", "_").replace("×","x").replace("/","-")
    q_onnx_ref = os.path.join(out_dir, f"q_{first_tag}.onnx")
    if os.path.exists(q_onnx_ref):
        q_model = onnx.load(q_onnx_ref)
        q_inits = {i.name: numpy_helper.to_array(i) for i in q_model.graph.initializer}
        print(f"{'Layer':<6} {'Op':<12} {'Output tensor':<40} {'SNR(dB)':>9} {'cosine':>8}")
        print("─" * 80)
        for i, node in enumerate(float_model.graph.node):
            if node.op_type not in ("Conv", "Gemm", "Add", "Relu", "MaxPool",
                                     "GlobalAveragePool", "Flatten"):
                continue
            out_name = node.output[0]
            float_t = float_env.get(out_name)
            flat_t = float_t
            if flat_t is not None:
                s_arr = q_inits.get(out_name + "_scale")
                if s_arr is None:
                    print(f"{i:<6} {node.op_type:<12} {out_name:<40} {'(no quant)':>9}")
                    continue
                s = float(s_arr.flat[0])
                z_arr = q_inits.get(out_name + "_zp")
                z = int(z_arr.flat[0]) if z_arr is not None else 0
                sim = sim_qdq(flat_t, s, z, signed=True)
                print(f"{i:<6} {node.op_type:<12} {out_name:<40} {snr_db(flat_t, sim):>8.1f}  {cosine(flat_t, sim):>8.4f}")


def main():
    import tempfile
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Float ONNX path (default: auto torchvision ResNet-18)")
    p.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 64, 64],
                   metavar=("N","C","H","W"))
    p.add_argument("--calib-n", type=int, default=4)
    p.add_argument("--out-dir", default="/tmp/coa_quant_report")
    args = p.parse_args()

    np.random.seed(42)
    shape = tuple(args.input_shape)

    if args.model is None:
        tmp = tempfile.mkdtemp()
        print("Exporting ResNet-18 from torchvision...")
        float_onnx = get_resnet18_onnx(tmp, shape)
    else:
        float_onnx = args.model

    run_report(float_onnx, shape, calib_n=args.calib_n, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
