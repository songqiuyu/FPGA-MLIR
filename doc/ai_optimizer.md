# AI 优化模块文档

## Phase 3-A：Tiling 自动搜索

### 问题定义

对于每个卷积层参数 `(N, M, R, C, k, s, d)`，在满足三个缓冲区硬件约束的前提下，
寻找使**缓冲区平均利用率最高**的分块参数 `(tM, tR, tC)`。

```
maximize  (wdepth/256 + gdepth/1024 + odepth/2048) / 3
subject to  wdepth < 256
            gdepth < 1024
            odepth < 2048
            tM ∈ {16, 32, ..., M}
            tR ∈ [1, R]
            tC ∈ [1, C]
```

### 方法对比

| 方法 | 文件 | 优势 | 局限 |
|---|---|---|---|
| 贪心（基线） | `old_version/tools/assign_addr.py` `get_tile()` | 极快（毫秒级） | 容易陷入局部最优 |
| 贝叶斯 TPE | `bayesian_opt.py` | 全局搜索，离散空间高效 | 需要多次评估（~200 trials） |
| DQN RL | `rl_agent.py` | 可泛化到分布外层形状 | 需要训练（~5000 步） |

### RL 环境规格（`env.py`）

```python
STATE_DIM  = 13   # [N/512, M/512, R/512, C/512, k/7, s/4, d/4,
                  #   wUtil, gUtil, oUtil, tM/M, tR/R, tC/C]
N_ACTIONS  = 9    # 0:tM/2  1:tM×2  2:nop  3:tR/2  4:tR×2
                  # 5:nop   6:tC/2  7:tC×2  8:submit
MAX_STEPS  = 50
REWARD     = utilization_score if legal else -1.0
```

### DQN 超参数

```python
GAMMA       = 0.99
LR          = 1e-3
BATCH_SIZE  = 64
BUFFER_SIZE = 10_000
EPS_DECAY   = 500   # steps
TARGET_SYNC = 50    # steps
```

---

## Phase 3-B：GNN 算子融合

### 计算图表示

```
节点特征（15维）：
  [0:5]   算子类型 one-hot（5类）
  [5]     M/512
  [6]     N/512
  [7]     R/512
  [8]     C/512
  [9]     tM/512
  [10]    tR/512
  [11]    tC/512
  [12]    in_scale（截断到 [0,1]）
  [13]    out_scale（截断到 [0,1]）
  [14]    |factor|/32768

边：有向数据流边（producer → consumer）
标签：1=可融合，0=不可融合
```

### FusionGAT 架构

```
x [N, 15]
  ↓ GATConv(15→64, heads=4, concat=True)  → ELU  [N, 256]
  ↓ GATConv(256→64, heads=1, concat=False) → ELU  [N, 64]
  ↓ EdgeMLP: cat(h_src, h_dst) [E, 128]
             Linear(128→64) → ReLU
             Linear(64→1)   → logit  [E]
  ↓ sigmoid > threshold → is_fusible [E]
```

### 可融合规则（训练数据标注）

```python
FUSIBLE_PAIRS = {
    "qlinearconv->qlinearadd",
    "qlinearconv->maxpool",
    "qlinearconv->qlinearglobalaveragepool",
    "qgemm->qlinearglobalaveragepool",
}
```

### 训练流程

```
1. 解析 old_version/parameters/*.mlir → 构建图列表
2. 按规则自动标注边标签
3. BCEWithLogitsLoss 训练 FusionGAT
4. 保存 fusion_gat.pth
5. 评估精确率 / 召回率
```

---

## Phase 3-C：AutoQuant 混合精度量化

### 整体流程

```
权重 .npz
  │
  ▼ sensitivity.py
per-layer 量化敏感度表 (sens[layer][bits])
  │
  ▼ pareto_search.py (NSGA-II)
Pareto 前沿：(acc_loss, hw_overhead, mem_pressure)
  │
  ▼ mixed_quant.py
混合精度 COA MLIR（in_scale/out_scale/factor 按层覆盖）
```

### 敏感度代理

```
sens(layer, bits) = ||W - Q(W, bits)||²_F
```

其中 `Q(W, bits)` 为对称均匀量化：
```
scale   = max(|W|) / (2^(bits-1) - 1)
Q(W, b) = round(W / scale) × scale
```

### NSGA-II 决策变量

```
基因 x[i] ∈ {0, 1, 2}  →  bits ∈ {4, 8, 16}
层数 n = 权重 tensor 数量

目标（均最小化）：
  f1 = Σ sens(i, bits[i]) / n_layers / max_sens   （精度损失代理）
  f2 = Σ bits[i]×params[i] / total_params / 16    （硬件开销）
  f3 = max(params[i] × bits[i] / 8) / 32MB        （内存峰值）
```

### scale 重量化公式

```python
# 当从 int8 切换到 bits 位时
new_scale = original_scale × (127 / (2^(bits-1) - 1))
new_factor = round(original_factor × (new_levels / 127)²)
```

---

## 模块间接口

```
coa-tiling (C++ Pass)
      ↑ 可由 tile_lookup.json 替换内置贪心逻辑

coa-op-fusion (C++ Pass)
      ↑ 可由 fusion_gat.pth 推理结果替换规则标注

COA MLIR 量化属性
      ↑ 可由 mixed_precision_config.json 覆盖
```

所有 AI 优化模块均**独立可运行**，不依赖 C++ 编译成功，以 Python 脚本形式提供接口。
