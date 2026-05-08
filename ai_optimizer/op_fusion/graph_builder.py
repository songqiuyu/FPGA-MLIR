"""
Op Fusion - Graph Builder (Phase 3-B)
Converts a COA MLIR text file into a PyTorch Geometric (PyG) graph
for GNN-based fusion pattern detection.

Node features (per op):
  [op_type_onehot(5), M, N, R, C, in_scale, out_scale, tM, tR, tC]  -> 15-dim

Edge: directed data-flow edges (producer -> consumer)

Node type encoding:
  0 = qlinearconv
  1 = qgemm
  2 = maxpool
  3 = qlinearadd
  4 = qlinearglobalaveragepool
"""

import re
import sys
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

# PyG is optional at import time; fail gracefully
try:
    import torch
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

OP_TYPE_MAP = {
    "qlinearconv":              0,
    "qgemm":                    1,
    "maxpool":                  2,
    "qlinearadd":               3,
    "qlinearglobalaveragepool": 4,
}
N_OP_TYPES = len(OP_TYPE_MAP)

# Total node feature dimension
NODE_FEAT_DIM = N_OP_TYPES + 10  # 5 onehot + 10 scalar features


def _onehot(idx: int, n: int) -> List[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _parse_attr_int(attrs: str, name: str, default: int = 0) -> int:
    m = re.search(rf'\b{name}\s*=\s*(0x[0-9a-fA-F]+|\d+)', attrs)
    if m:
        val = m.group(1)
        return int(val, 16) if val.startswith("0x") else int(val)
    return default


def _parse_attr_float(attrs: str, name: str, default: float = 1.0) -> float:
    m = re.search(rf'\b{name}\s*=\s*([\d.eE+\-]+)', attrs)
    return float(m.group(1)) if m else default


def build_graph_from_mlir(mlir_path: str) -> Optional[Dict]:
    """
    Parse a COA MLIR file and return a graph dict:
      {
        "node_features": np.ndarray [N_nodes, NODE_FEAT_DIM],
        "edge_index":    np.ndarray [2, N_edges]  (src, dst),
        "op_names":      List[str],
        "op_types":      List[int],
        "mlir_ops":      List[dict]  (raw parsed attributes per node)
      }
    """
    with open(mlir_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r'%(\w+)\s*=\s*"coa\.(\w+)"\(([^)]*)\)\s*\{([^}]*)\}'
    matches = list(re.finditer(pattern, content))

    if not matches:
        return None

    # Map output SSA name -> node index
    name_to_idx: Dict[str, int] = {}
    nodes: List[Dict] = []

    for match in matches:
        out_name = match.group(1)
        op_name  = match.group(2)
        args_str = match.group(3)
        attrs    = match.group(4)

        in_vars = [v.strip().lstrip("%") for v in args_str.split(",") if v.strip()]
        op_type = OP_TYPE_MAP.get(op_name, -1)
        if op_type < 0:
            continue  # skip unknown ops

        idx = len(nodes)
        name_to_idx[out_name] = idx

        # Extract scalar features (normalised)
        M       = _parse_attr_int(attrs, "M")
        N       = _parse_attr_int(attrs, "N")
        R       = _parse_attr_int(attrs, "R")
        C       = _parse_attr_int(attrs, "C")
        tM      = _parse_attr_int(attrs, "tM")
        tR      = _parse_attr_int(attrs, "tR")
        tC      = _parse_attr_int(attrs, "tC")
        in_s    = _parse_attr_float(attrs, "in_scale", 1.0)
        out_s   = _parse_attr_float(attrs, "out_scale", 1.0)
        factor  = _parse_attr_int(attrs, "factor")

        nodes.append({
            "out_name": out_name,
            "op_name":  op_name,
            "op_type":  op_type,
            "in_vars":  in_vars,
            "M": M, "N": N, "R": R, "C": C,
            "tM": tM, "tR": tR, "tC": tC,
            "in_scale": in_s, "out_scale": out_s, "factor": factor,
        })

    if not nodes:
        return None

    # Build node feature matrix
    max_dim = 512.0
    node_features = []
    for nd in nodes:
        feats = _onehot(nd["op_type"], N_OP_TYPES) + [
            nd["M"]  / max_dim,
            nd["N"]  / max_dim,
            nd["R"]  / max_dim,
            nd["C"]  / max_dim,
            nd["tM"] / max_dim,
            nd["tR"] / max_dim,
            nd["tC"] / max_dim,
            min(nd["in_scale"],  1.0),
            min(nd["out_scale"], 1.0),
            min(abs(nd["factor"]) / 32768.0, 1.0),
        ]
        node_features.append(feats)

    # Build edges: for each node, connect its in_var producers -> this node
    edges_src, edges_dst = [], []
    for i, nd in enumerate(nodes):
        for var in nd["in_vars"]:
            if var in name_to_idx:
                edges_src.append(name_to_idx[var])
                edges_dst.append(i)

    return {
        "node_features": np.array(node_features, dtype=np.float32),
        "edge_index":    np.array([edges_src, edges_dst], dtype=np.int64),
        "op_names":      [nd["out_name"] for nd in nodes],
        "op_types":      [nd["op_type"]  for nd in nodes],
        "mlir_ops":      nodes,
    }


def to_pyg_data(graph: Dict):
    """Convert graph dict to a PyTorch Geometric Data object."""
    if not _HAS_PYG:
        raise ImportError("torch_geometric not installed: pip install torch-geometric")
    import torch
    x  = torch.tensor(graph["node_features"], dtype=torch.float)
    ei = torch.tensor(graph["edge_index"],    dtype=torch.long)
    return Data(x=x, edge_index=ei)


def enumerate_candidate_pairs(graph: Dict) -> List[Tuple[int, int, str]]:
    """
    Return all directed (producer, consumer) pairs as fusion candidates.
    Each tuple: (src_idx, dst_idx, "op_src->op_dst")
    """
    pairs = []
    edge_index = graph["edge_index"]
    op_names   = graph["op_names"]
    op_types   = graph["op_types"]
    ot_inv     = {v: k for k, v in OP_TYPE_MAP.items()}

    if edge_index.shape[1] == 0:
        return pairs

    for k in range(edge_index.shape[1]):
        src = int(edge_index[0, k])
        dst = int(edge_index[1, k])
        label = f"{ot_inv.get(op_types[src],'?')}->{ot_inv.get(op_types[dst],'?')}"
        pairs.append((src, dst, label))
    return pairs
