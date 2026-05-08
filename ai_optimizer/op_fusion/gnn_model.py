"""
Op Fusion - GNN Classifier (Phase 3-B)
A Graph Attention Network (GAT) that classifies each directed (src, dst) edge
in the COA compute graph as fusible (1) or non-fusible (0).

Architecture:
  Input: node features [N, NODE_FEAT_DIM]
  GATConv × 2 -> edge MLP -> binary classification

Requirements: torch, torch-geometric
  pip install torch torch-geometric
"""

import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np

NODE_FEAT_DIM = 15  # from graph_builder.py
HIDDEN_DIM    = 64
N_HEADS       = 4

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, DataLoader
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    print("[gnn_model] torch-geometric not found. Install: pip install torch torch-geometric")


# ---- Known fusible op-pair labels (from rule-based expertise) ----
FUSIBLE_PAIRS = {
    "qlinearconv->qlinearadd",           # residual add after conv
    "qlinearconv->maxpool",              # conv -> pool chain
    "qlinearconv->qlinearglobalaveragepool",
    "qgemm->qlinearglobalaveragepool",
}

NONFUSIBLE_PAIRS = {
    "maxpool->qlinearadd",
    "qlinearadd->qlinearconv",
    "qlinearglobalaveragepool->qlinearconv",
}


def label_pair(src_op: str, dst_op: str) -> int:
    """Rule-based label for training data generation (1=fusible, 0=not)."""
    key = f"{src_op}->{dst_op}"
    if key in FUSIBLE_PAIRS:
        return 1
    if key in NONFUSIBLE_PAIRS:
        return 0
    return 0  # conservative default


class FusionGAT(nn.Module):
    """
    GAT-based edge classifier for op fusion.

    Forward returns per-edge logits (fusible probability).
    """

    def __init__(self, node_dim: int = NODE_FEAT_DIM,
                 hidden: int = HIDDEN_DIM, heads: int = N_HEADS):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch-geometric required")

        self.gat1 = GATConv(node_dim,   hidden,          heads=heads, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden,      heads=1,     concat=False)

        # Edge MLP: concatenate (src_emb, dst_emb) -> binary logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index):
        """
        Args:
            x:          [N, node_dim] node features
            edge_index: [2, E] directed edges

        Returns:
            logits: [E] per-edge fusion logit
        """
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))   # [N, hidden]

        src, dst = edge_index
        edge_feats = torch.cat([h[src], h[dst]], dim=-1)  # [E, 2*hidden]
        logits = self.edge_mlp(edge_feats).squeeze(-1)     # [E]
        return logits

    def predict_fusible(self, x, edge_index, threshold: float = 0.5):
        """Returns bool tensor [E] indicating which edges are fusible."""
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return torch.sigmoid(logits) > threshold


def build_training_sample(graph: Dict):
    """
    Build (Data, labels) from a parsed COA graph dict.
    Labels are assigned by rule-based heuristic for bootstrapping.
    """
    if not _HAS_PYG:
        return None, None

    import torch
    from graph_builder import OP_TYPE_MAP

    x  = torch.tensor(graph["node_features"], dtype=torch.float)
    ei = torch.tensor(graph["edge_index"],    dtype=torch.long)
    op_types   = graph["op_types"]
    op_type_inv = {v: k for k, v in OP_TYPE_MAP.items()}

    # Build edge labels
    labels = []
    for k in range(ei.shape[1]):
        src_op = op_type_inv.get(op_types[ei[0, k].item()], "?")
        dst_op = op_type_inv.get(op_types[ei[1, k].item()], "?")
        labels.append(label_pair(src_op, dst_op))

    y = torch.tensor(labels, dtype=torch.float)
    data = Data(x=x, edge_index=ei, y=y)
    return data, y


def train_gnn(graphs: List[Dict],
              epochs: int = 50,
              lr: float = 1e-3,
              save_path: str = "fusion_gat.pth") -> Optional["FusionGAT"]:
    """
    Train the GNN fusion classifier on a list of COA graph dicts.
    Returns the trained model (or None if PyG unavailable).
    """
    if not _HAS_PYG:
        print("[gnn_model] Skipping training: torch-geometric not available.")
        return None

    import torch

    model     = FusionGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    # Build dataset
    dataset = []
    for g in graphs:
        data, _ = build_training_sample(g)
        if data is not None and data.edge_index.shape[1] > 0:
            dataset.append(data)

    if not dataset:
        print("[gnn_model] No valid training samples found.")
        return None

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for data in dataset:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss   = loss_fn(logits, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[gnn_model] Model saved to {save_path}")
    return model


def load_gnn(save_path: str = "fusion_gat.pth") -> Optional["FusionGAT"]:
    """Load a previously trained FusionGAT model."""
    if not _HAS_PYG or not os.path.exists(save_path):
        return None
    import torch
    model = FusionGAT()
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    model.eval()
    return model


def predict_fusion(model: "FusionGAT", graph: Dict,
                   threshold: float = 0.5) -> List[Tuple[int, int, bool]]:
    """
    Run GNN inference on a graph dict and return per-edge fusion decisions.
    Returns list of (src_idx, dst_idx, is_fusible).
    """
    if model is None or not _HAS_PYG:
        return []

    import torch
    x  = torch.tensor(graph["node_features"], dtype=torch.float)
    ei = torch.tensor(graph["edge_index"],    dtype=torch.long)

    decisions = model.predict_fusible(x, ei, threshold)
    result = []
    for k in range(ei.shape[1]):
        src = int(ei[0, k].item())
        dst = int(ei[1, k].item())
        fuse = bool(decisions[k].item())
        result.append((src, dst, fuse))
    return result
