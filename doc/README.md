# FPGA-MLIR AI 编译器 — 项目文档

> 面向自定义 FPGA 粗粒度 VLIW 指令集的 AI 编译器，兼具学术研究价值与工程落地价值。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [COA 方言（C++ MLIR）](#4-coa-方言c-mlir)
5. [编译器 Pass 流水线](#5-编译器-pass-流水线)
6. [VLIW 指令结构](#6-vliw-指令结构)
7. [AI 优化模块](#7-ai-优化模块)
8. [测试体系](#8-测试体系)
9. [构建说明](#9-构建说明)
10. [使用示例](#10-使用示例)
11. [路线图](#11-路线图)

---

## 1. 项目概述

本项目为一款针对自定义 FPGA 粗粒度加速器的端到端 AI 编译器，输入 ONNX 量化模型，输出可直接由 FPGA DMA 加载的 512-bit VLIW 二进制指令流。

**两大目标**：

| 目标 | 说明 |
|---|---|
| **AI-oriented**（面向 AI） | 将量化神经网络编译到 FPGA VLIW 指令集，覆盖图优化、量化、Tiling、地址分配、代码生成完整流水线 |
| **AI-empowered**（AI 赋能） | 用强化学习 / 贝叶斯优化 / GNN 自动化编译器关键决策（Tiling 搜索、算子融合、混合精度量化） |

**硬件目标**：
- 512-bit（64 字节）VLIW 指令，每条指令驱动一次粗粒度计算（卷积 / 池化 / 全局平均池化 / 加法）
- INT8 量化激活与权重，per-channel 量化
- 片上缓冲区约束：`wdepth < 256`，`gdepth < 1024`，`odepth < 2048`

---

## 2. 整体架构

```
ONNX 模型
    │
    ▼
[ONNX Importer]  (old_version/coa_mlir/frontend/importer.py)
    │  生成高层 COA MLIR
    ▼
┌──────────────────────────────────────────────────────┐
│               COA MLIR 编译器流水线                   │
│                                                      │
│  coa-shape-infer  →  填充 R,C,M,N,R0,C0             │
│  coa-op-fusion    →  Conv+Add / Conv+Activation 融合 │
│  coa-tiling       →  计算 tM,tR,tC（缓冲区感知）    │
│  coa-addr-assign  →  DDR 地址 + 量化因子            │
│  coa-legalize     →  VLIW 位域合法性验证            │
│  coa-vliw-gen     →  生成 512-bit VLIW 二进制       │
└──────────────────────────────────────────────────────┘
    │
    ▼
VLIW Binary (.vliw)  ──→  FPGA DMA 加载
```

AI 优化模块作为**可替换组件**插入流水线：

```
coa-tiling  ←→  ai_optimizer/tiling_search/  (RL / 贝叶斯)
coa-op-fusion ←→  ai_optimizer/op_fusion/    (GNN)
量化参数    ←→  ai_optimizer/auto_quant/     (NSGA-II)
```

---

## 3. 目录结构

```
FPGA-MLIR/
├── coa_compiler/                  # C++ MLIR 编译器（新）
│   ├── include/COA/
│   │   ├── COADialect.td          # 方言 TableGen ODS
│   │   ├── COAOps.td              # 5 个算子定义
│   │   ├── COAPasses.td           # 6 个 Pass 定义
│   │   ├── COADialect.h           # 方言头文件
│   │   ├── COAOps.h               # 算子头文件
│   │   ├── COAPasses.h            # Pass 头文件
│   │   └── VLIWDefs.h             # 512-bit VLIW 结构体
│   ├── lib/
│   │   ├── Dialect/               # COADialect.cpp / COAOps.cpp
│   │   ├── Transforms/            # 6 个 Pass 实现 + RegisterPasses
│   │   └── CodeGen/               # VLIWCodeGen.cpp（位域打包）
│   ├── tools/
│   │   ├── coa-opt/               # mlir-opt 风格调试工具
│   │   └── coa-compiler/          # 一键 MLIR→VLIW 编译器
│   ├── CMakeLists.txt             # 根 CMake
│   └── build.bat                  # Windows 构建脚本
│
├── ai_optimizer/                  # AI 优化模块（新）
│   ├── tiling_search/
│   │   ├── env.py                 # Gym-compatible RL 环境
│   │   ├── bayesian_opt.py        # 贝叶斯优化（Optuna TPE）
│   │   └── rl_agent.py            # DQN 强化学习智能体
│   ├── op_fusion/
│   │   ├── graph_builder.py       # MLIR → PyG 计算图
│   │   ├── gnn_model.py           # FusionGAT（边分类器）
│   │   └── train_fusion.py        # GNN 训练脚本
│   ├── auto_quant/
│   │   ├── sensitivity.py         # Fisher 量化敏感度分析
│   │   ├── pareto_search.py       # NSGA-II 多目标 Pareto 搜索
│   │   └── mixed_quant.py         # 混合精度 MLIR 重写
│   └── requirements.txt
│
├── tests/                         # 新测试（新）
│   ├── test_vliw_codegen.py       # VLIW 位打包 + 解析 + 约束测试
│   └── test_ai_optimizer.py       # AI 优化模块单元测试
│
├── old_version/                   # 原 Python 参考流水线（存档）
│   ├── tools/                     # Python 参考工具（vliw.py 等）
│   ├── coa_mlir/                  # Python MLIR 前端
│   ├── c_reference/               # C 参考推理实现
│   ├── parameters/                # 生成的 MLIR / 参数文件
│   ├── models/                    # ONNX 模型
│   ├── datasets/                  # 标定数据集
│   └── tests/                     # 旧测试
│
├── doc/                           # 项目文档（本文件夹）
├── llvm-project/                  # LLVM 源码（gitignored）
└── .gitignore
```

---

## 4. COA 方言（C++ MLIR）

### 方言信息

| 属性 | 值 |
|---|---|
| 方言名 | `coa` |
| C++ 命名空间 | `mlir::coa` |
| TableGen 定义 | `include/COA/COADialect.td` |

### 支持的算子

| 算子 | VLIW op 类型 | 说明 |
|---|---|---|
| `coa.qlinearconv` | 0 (Conv) | INT8 量化 2D 卷积，支持 per-channel 权重量化 |
| `coa.qgemm` | 0 (Conv) | INT8 量化全连接层（GEMM），映射为 1×1 卷积 |
| `coa.maxpool` | 1 (Pool) | 整数最大池化 |
| `coa.qlinearglobalaveragepool` | 2 (GAP) | 量化全局平均池化 |
| `coa.qlinearadd` | 3 (Add) | 量化元素加法（残差连接） |

### 属性层级

每个算子的属性分为三层，由不同 Pass 逐层填入：

```
Level 1  ONNX 层（必填，导入时填入）
  in_scale / in_zp / weight_scale / weight_zp / out_scale / out_zp
  kernel_shape / strides / pads / dilations

Level 2  形状层（由 --coa-shape-infer 填入）
  R / C / M / N / R0 / C0 / sM_concat / M_concat

Level 3  硬件层（由 --coa-tiling 和 --coa-addr-assign 填入）
  tM / tR / tC          ← 分块参数
  in_addr / out_addr / weight_addr / bias_addr / silu_addr  ← DDR 地址
  factor / factor2       ← 定点量化系数
```

---

## 5. 编译器 Pass 流水线

### Pass 顺序

```
--coa-shape-infer
    └─ 推断所有算子的 R,C,M,N,R0,C0
       依据：输入 tensor 形状 + kernel/stride/pad/dilation

--coa-op-fusion
    └─ Conv+Add（残差）模式标记 silu_addr
       AI 替换点：ai_optimizer/op_fusion/gnn_model.py

--coa-tiling
    └─ 贪心求解最大合法 (tM,tR,tC)
       约束：wdepth < 256, gdepth < 1024, odepth < 2048
       AI 替换点：ai_optimizer/tiling_search/

--coa-addr-assign
    └─ 分配 DDR 地址（Ping-Pong 激活缓冲）
       计算量化系数 factor = round(s_in * s_w / s_out * 2^15)

--coa-legalize
    └─ 验证所有位域不溢出（11/12/34/36 bit 上界检查）
       失败则 signalPassFailure() 并打印诊断

--coa-vliw-gen
    └─ 按算子顺序打包 512-bit VLIW
       输出：N × 64 字节二进制文件
```

### Pass 选项示例

```bash
coa-opt model.mlir \
  --coa-shape-infer \
  --coa-op-fusion \
  --coa-tiling="wdepth-limit=256 gdepth-limit=1024 odepth-limit=2048" \
  --coa-addr-assign="weight-base=0x8000000 bias-base=0xC0000000" \
  --coa-legalize \
  --coa-vliw-gen="output=model.vliw"
```

---

## 6. VLIW 指令结构

512-bit（64 字节）指令，LSB-first 位域布局：

| 位段 | 宽度 | 字段名 | 说明 |
|---|---|---|---|
| [7:0] | 8 | operator | 0=Conv, 1=Pool, 2=GAP, 3=Add |
| [43:8] | 36 | DDR_x1_address | 输入激活 DDR 地址 |
| [79:44] | 36 | DDR_x2_address | 权重 DDR 地址 |
| [90:80] | 11 | Bias_source_address | Bias DDR 地址 |
| [126:91] | 36 | Result_dest_address | 输出 DDR 地址 |
| [134:127] | 8 | Activate_LUT_address | 激活 LUT 页地址 |
| [145:135] | 11 | R | 输出高度（max 2047） |
| [156:146] | 11 | C | 输出宽度 |
| [168:157] | 12 | M | 输出通道数（max 4095） |
| [180:169] | 12 | N | 输入通道数 |
| [191:181] | 11 | R0 | 输入高度 |
| [202:192] | 11 | C0 | 输入宽度 |
| [214:203] | 12 | sM_concat | Concat 通道偏移 |
| [226:215] | 12 | M_concat | Concat 通道数 |
| [234:227] | 8 | Quant_x1_z | 输入零点 |
| [242:235] | 8 | Quant_x2_z | 权重零点 |
| [250:243] | 8 | Quant_y_z | 输出零点 |
| [253:251] | 3 | Conv_pad | 填充 |
| [258:254] | 5 | Conv_kernel | 卷积核大小 |
| [261:259] | 3 | Conv_stride | 步长 |
| [264:262] | 3 | Conv_dilation | 膨胀率 |
| [275:265] | 11 | Conv_tR | Tile 行数 |
| [286:276] | 11 | Conv_tC | Tile 列数 |
| [298:287] | 12 | Conv_tM | Tile 输出通道 |
| [310:299] | 12 | Conv_tN | Tile 输入通道 |
| [318:311] | 8 | Conv_permute* | 排列标志（保留） |
| [352:319] | 34 | Conv_quant_factor | 量化系数（有符号） |
| [384:353] | 32 | Conv_quant_factor2 | Add 量化系数 2 |
| [511:385] | 127 | — | 保留（清零） |

**量化系数计算**：
```
factor  = round(in_scale × weight_scale / out_scale × 2^15)    # Conv/GEMM
factor  = round(a_scale  / out_scale × 2^15)                   # Add (输入A)
factor2 = round(b_scale  / out_scale × 2^15)                   # Add (输入B)
```

---

## 7. AI 优化模块

### Phase 3-A：Tiling 自动搜索

#### 方法一：贝叶斯优化（`bayesian_opt.py`）

使用 **Optuna TPE 采样器**在离散空间 `(tM ∈ {16,32,...,M}, tR ∈ [1,R], tC ∈ [1,C])` 中最大化缓冲区利用率目标：

```
score = (wdepth/wLimit + gdepth/gLimit + odepth/oLimit) / 3
约束：所有利用率 ≤ 1.0
```

输出 `tile_lookup.json`，可直接替换 `--coa-tiling` 的贪心逻辑。

#### 方法二：DQN 强化学习（`rl_agent.py`）

- **状态**（13维）：层参数归一化 + 当前 tile 缓冲区利用率
- **动作**（9种）：tM/tR/tC 的二分之一缩放 / 两倍扩展 / 提交
- **奖励**：合法时返回均值利用率，违约返回 -1
- **训练**：在随机生成的层参数分布上训练 5000 步

```
# 与贪心基线对比
python ai_optimizer/tiling_search/rl_agent.py
```

### Phase 3-B：GNN 算子融合

#### 图构建（`graph_builder.py`）

将 COA MLIR 文件解析为 PyTorch Geometric `Data` 对象：
- **节点**：每个 COA op，15 维特征（5维 one-hot 类型 + 10维标量）
- **边**：数据流方向（producer → consumer）

#### FusionGAT 模型（`gnn_model.py`）

```
输入: 节点特征 [N, 15]
GATConv(15 → 64, heads=4)   → ELU
GATConv(256 → 64, heads=1)  → ELU
边 MLP: cat(src, dst) → [128] → ReLU → [1]  (logit)
输出: 每条边的融合概率
```

已知可融合模式（规则标注用于引导训练）：

| 模式 | 说明 |
|---|---|
| `qlinearconv → qlinearadd` | 残差加法 |
| `qlinearconv → maxpool` | Conv-Pool 链 |
| `qlinearconv → qlinearglobalaveragepool` | 末尾 GAP |

#### 训练

```bash
python ai_optimizer/op_fusion/train_fusion.py \
    --mlir-dir old_version/parameters \
    --epochs 100
```

### Phase 3-C：AutoQuant 混合精度量化

#### 敏感度分析（`sensitivity.py`）

基于 **Fisher 信息对角近似（HAWQ 风格）**，逐层计算量化误差代理：

```
sensitivity[bits] = ||W - Q(W, bits)||²_F
```

层按 4-bit 误差从高到低排列，敏感度高的层使用更多比特。

#### NSGA-II Pareto 搜索（`pareto_search.py`）

三目标同时最优化（均为最小化）：

| 目标 | 计算 | 含义 |
|---|---|---|
| f₁ 精度损失代理 | Fisher 误差加权和 | 量化对精度的影响 |
| f₂ 硬件开销 | 均值比特宽度 / 16 | 指令/内存开销 |
| f₃ 缓冲峰值压力 | 最大层权重字节 / 32MB | 片上内存压力 |

输出 Pareto 前沿 JSON，选取精度最优配置写入 `mixed_precision_config.json`。

#### 应用（`mixed_quant.py`）

```bash
python ai_optimizer/auto_quant/mixed_quant.py \
    --config mixed_precision_config.json \
    --mlir   model_assigned.mlir \
    --output model_mixed.mlir
```

---

## 8. 测试体系

```bash
# 运行全部测试
G:\Anaconda3\envs\502\python.exe -m pytest tests/ -v
```

**当前状态：26/26 全部通过 ✅**

| 测试类 | 文件 | 覆盖内容 |
|---|---|---|
| `TestPythonVLIWPacking` | `test_vliw_codegen.py` | VLIW 位域打包正确性（8 tests） |
| `TestExtractVLIW` | `test_vliw_codegen.py` | MLIR 解析流程（1 test） |
| `TestTilingConstraints` | `test_vliw_codegen.py` | 缓冲区约束验证（3 tests） |
| `TestBayesianTiling` | `test_vliw_codegen.py` | 贝叶斯搜索合法性（1 test） |
| `TestTilingEnv` | `test_ai_optimizer.py` | RL 环境状态/动作/奖励（5 tests） |
| `TestGraphBuilder` | `test_ai_optimizer.py` | GNN 图构建（3 tests） |
| `TestSensitivity` | `test_ai_optimizer.py` | 量化敏感度计算（3 tests） |
| `TestMixedQuant` | `test_ai_optimizer.py` | 混合精度 MLIR 重写（2 tests） |

---

## 9. 构建说明

### 前提条件

1. **构建 llvm-project**（约 30-60 分钟，需要 ~20GB 磁盘）：

```bat
cmake -S llvm-project\llvm -B llvm-project\build ^
      -G "Visual Studio 17 2022" -A x64 ^
      -DLLVM_ENABLE_PROJECTS="mlir;clang" ^
      -DLLVM_TARGETS_TO_BUILD="X86" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build llvm-project\build --config Release -j8
```

2. **构建 COA 编译器**：

```bat
cd coa_compiler
build.bat
```

生成产物：
- `coa_compiler\build\bin\Release\coa-opt.exe`
- `coa_compiler\build\bin\Release\coa-compiler.exe`

### Python AI 优化模块依赖

```bash
pip install -r ai_optimizer/requirements.txt
# 核心：numpy onnx
# Phase 3-A: optuna torch
# Phase 3-B: torch torch-geometric
# Phase 3-C: pymoo
```

---

## 10. 使用示例

### 逐步调试（coa-opt）

```bash
# 仅运行形状推断
coa-opt --coa-shape-infer model.mlir -o model_shape.mlir

# 运行完整流水线（不含代码生成）
coa-opt --coa-shape-infer --coa-op-fusion \
        --coa-tiling --coa-addr-assign --coa-legalize \
        model.mlir -o model_lowered.mlir

# 完整编译到 VLIW
coa-opt --coa-shape-infer --coa-op-fusion \
        --coa-tiling --coa-addr-assign --coa-legalize \
        --coa-vliw-gen="output=model.vliw" \
        model.mlir
```

### 一键编译（coa-compiler）

```bash
coa-compiler --output model.vliw \
             --weight-base 0x8000000 \
             --bias-base 0xC0000000 \
             --act-base 0x10000000 \
             model.mlir
```

### AI Tiling 搜索

```python
# 贝叶斯优化
from ai_optimizer.tiling_search.bayesian_opt import bayesian_tile_search
tM, tR, tC = bayesian_tile_search(N=64, M=256, R=56, C=56, k=3, s=1, d=1)

# 构建查找表
python ai_optimizer/tiling_search/bayesian_opt.py
# → 生成 tile_lookup.json
```

### AutoQuant

```bash
# 1. 分析敏感度
python ai_optimizer/auto_quant/sensitivity.py model_weights.npz

# 2. Pareto 搜索
python ai_optimizer/auto_quant/pareto_search.py model_weights.npz 100 50
# → 生成 pareto_front.json, mixed_precision_config.json

# 3. 应用到 MLIR
python ai_optimizer/auto_quant/mixed_quant.py \
    --config mixed_precision_config.json \
    --mlir model_assigned.mlir \
    --output model_mixed.mlir
```

---

## 11. 路线图

| 阶段 | 状态 | 内容 |
|---|---|---|
| Phase 1-A | ✅ 完成 | COA Dialect TableGen ODS + CMake 构建系统 |
| Phase 1-B | ✅ 完成 | 6 个 MLIR Pass 实现（C++） |
| Phase 1-C/D | ✅ 完成 | VLIW 代码生成后端 + coa-opt / coa-compiler 工具 |
| Phase 2 | ✅ 完成 | 基准测试体系（26 tests passing） |
| Phase 3-A | ✅ 完成 | Tiling 自动搜索（贝叶斯 + DQN） |
| Phase 3-B | ✅ 完成 | GNN 算子融合（FusionGAT） |
| Phase 3-C | ✅ 完成 | AutoQuant 混合精度量化搜索（NSGA-II） |
| **M8** | 🔲 待完成 | 构建 llvm-project，端到端验证 C++ coa-opt / coa-compiler |
| M9 | 🔲 计划中 | ONNX → COA MLIR 前端桥接（替换 Python importer） |
| M10 | 🔲 计划中 | ResNet-18 / YOLO10 全模型端到端 VLIW 正确性验证 |
| M11 | 🔲 计划中 | 学术论文：AI-Empowered MLIR Compiler for FPGA VLIW |
