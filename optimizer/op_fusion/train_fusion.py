"""
Op Fusion - GNN Training Script (Phase 3-B)

Usage:
  python train_fusion.py --mlir-dir <dir_with_mlir_files> [--epochs 100]

Loads all .mlir files, builds graph dicts, labels edges by rule-based heuristic,
trains FusionGAT, saves model to fusion_gat.pth, and prints per-edge accuracy.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from graph_builder import build_graph_from_mlir, enumerate_candidate_pairs
from gnn_model import train_gnn, load_gnn, predict_fusion, label_pair, FUSIBLE_PAIRS

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def collect_mlir_graphs(mlir_dir: str):
    """Walk mlir_dir and parse all *.mlir files into graph dicts."""
    paths = glob.glob(os.path.join(mlir_dir, "**", "*.mlir"), recursive=True)
    graphs = []
    for p in paths:
        g = build_graph_from_mlir(p)
        if g and g["edge_index"].shape[1] > 0:
            graphs.append(g)
            print(f"  Loaded {p}: {len(g['op_names'])} ops, "
                  f"{g['edge_index'].shape[1]} edges")
    return graphs


def evaluate_model(model, graphs):
    """Compute per-edge accuracy of the trained model vs rule labels."""
    if model is None:
        return
    from gnn_model import OP_TYPE_MAP
    ot_inv = {v: k for k, v in OP_TYPE_MAP.items()}

    correct = total = 0
    tp = fp = tn = fn = 0

    for g in graphs:
        preds = predict_fusion(model, g, threshold=0.5)
        op_types = g["op_types"]
        for src, dst, is_fuse in preds:
            src_op = ot_inv.get(op_types[src], "?")
            dst_op = ot_inv.get(op_types[dst], "?")
            gt = label_pair(src_op, dst_op)
            pred = int(is_fuse)
            correct += (pred == gt)
            total += 1
            if gt == 1 and pred == 1: tp += 1
            if gt == 0 and pred == 1: fp += 1
            if gt == 0 and pred == 0: tn += 1
            if gt == 1 and pred == 0: fn += 1

    if total > 0:
        acc = correct / total * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        print(f"\n  Accuracy: {acc:.1f}%  Precision: {prec:.1f}%  Recall: {rec:.1f}%")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")


def main():
    parser = argparse.ArgumentParser(description="Train GNN op-fusion classifier")
    parser.add_argument("--mlir-dir", default="../../examples/resnet18/model",
                        help="Directory containing COA .mlir files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--save",   default="fusion_gat.pth")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    mlir_dir = os.path.join(os.path.dirname(__file__), args.mlir_dir)
    print(f"=== COA Op Fusion GNN Training ===")
    print(f"MLIR dir: {mlir_dir}")
    print(f"Known fusible patterns: {sorted(FUSIBLE_PAIRS)}\n")

    graphs = collect_mlir_graphs(mlir_dir)
    if not graphs:
        print("[train_fusion] No .mlir files found. Generating synthetic samples...")
        # Create a minimal synthetic graph for demonstration
        import numpy as np
        graphs = [_make_synthetic_graph()]

    if args.eval_only:
        model = load_gnn(args.save)
        if model is None:
            print("[train_fusion] No saved model found at", args.save)
            return
    else:
        model = train_gnn(graphs, epochs=args.epochs, lr=args.lr,
                          save_path=args.save)

    print("\n=== Evaluation ===")
    evaluate_model(model, graphs)


def _make_synthetic_graph():
    """Minimal synthetic ResNet block graph for smoke testing."""
    import numpy as np
    # Simulate: conv -> add -> conv -> pool
    # Nodes: 4 ops, edges: conv->add, add->conv, conv->pool
    n_nodes = 4
    node_features = np.zeros((n_nodes, 15), dtype=np.float32)
    # qlinearconv=0, qlinearadd=3, maxpool=2
    op_types = [0, 3, 0, 2]
    for i, ot in enumerate(op_types):
        node_features[i, ot] = 1.0  # onehot
        node_features[i, 5]  = 0.25  # M/512
        node_features[i, 6]  = 0.25  # N/512
        node_features[i, 7]  = 0.1   # R/512
        node_features[i, 8]  = 0.1   # C/512

    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    return {
        "node_features": node_features,
        "edge_index":    edge_index,
        "op_names":      ["conv1", "add1", "conv2", "pool1"],
        "op_types":      op_types,
        "mlir_ops":      [],
    }


if __name__ == "__main__":
    main()
