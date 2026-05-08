# FPGA-MLIR AI 编译器 — 项目文档

> 面向自定义 FPGA 粗粒度 VLIW 指令集的 AI 编译器，兼具学术研究价值与工程落地价值。

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [各程序功能详解](#4-各程序功能详解)
5. [COA 方言（MLIR 算子定义）](#5-coa-方言mlir-算子定义)
6. [编译器 Pass 流水线](#6-编译器-pass-流水线)
7. [VLIW 指令结构](#7-vliw-指令结构)
8. [AI 优化模块](#8-ai-优化模块)
9. [测试体系](#9-测试体系)
10. [构建说明](#10-构建说明)
11. [完整使用示例](#11-完整使用示例)
12. [路线图](#12-路线图)

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

项目数据流从 PyTorch/ONNX 浮点模型出发，经过量化、格式转换、多级编译，最终产生供 FPGA DMA 直接加载的二进制指令流。

```
┌─────────────────────────────────────────────────────────────┐
│                     用户输入                                 │
│  ResNet-18 / YOLO / ...  (PyTorch 浮点模型)                 │
└────────────────────┬────────────────────────────────────────┘
                     │  export_onnx.py (PTQ 量化)
                     ▼
              resnet18_quant_int8.onnx
                     │  import_mlir.py / coa.onnx_importer
                     ▼
         ┌───────────────────────┐
         │  COA MLIR  (Level 1)  │  ← ONNX 量化参数 + 卷积几何
         │  coa.qlinearconv      │    in_scale, weight_scale,
         │  coa.maxpool          │    kernel_shape, strides …
         │  coa.qgemm            │
         └──────────┬────────────┘
                    │
         ┌──────────▼──────────────────────────────────────────┐
         │           C++ MLIR 编译器流水线 (6 Passes)           │
         │                                                      │
         │  1. coa-shape-infer  → 填充 R,C,M,N,R0,C0           │
         │  2. coa-op-fusion    → Conv+Add 残差融合             │
         │  3. coa-tiling       → 计算 tM,tR,tC (缓冲区感知)   │
         │  4. coa-addr-assign  → DDR 地址 + 量化因子 factor    │
         │  5. coa-legalize     → 全 VLIW 位域合法性验证        │
         │  6. coa-vliw-gen     → 生成 N×512-bit 二进制流       │
         └──────────┬──────────────────────────────────────────┘
                    │
                    ▼
           resnet18.vliw  (N×64 字节)
                    │
                    ▼
            FPGA DMA 加载 → 硬件执行
```

AI 优化模块以**可替换组件**的形式插入流水线关键决策点：

```
Pass 3 coa-tiling     ←→  optimizer/tiling_search/  (贝叶斯 TPE / DQN RL)
Pass 2 coa-op-fusion  ←→  optimizer/op_fusion/      (FusionGAT 图神经网络)
ONNX 量化参数         ←→  optimizer/auto_quant/     (NSGA-II Pareto 搜索)
```

---

## 3. 目录结构

```
FPGA-MLIR/
├── compiler/                      # C++ MLIR 编译器
│   ├── include/COA/
│   │   ├── COADialect.td          # 方言 TableGen ODS 定义
│   │   ├── COAOps.td              # 5 个算子的完整属性定义
│   │   ├── COAPasses.td           # 6 个 Pass 的 TableGen 定义
│   │   ├── COADialect.h/.cpp      # 方言初始化与属性解析
│   │   ├── COAOps.h               # 算子类头文件（TableGen 生成）
│   │   ├── COAPasses.h            # Pass 工厂函数与注册
│   │   └── VLIWDefs.h             # 512-bit VLIW C++ 结构体
│   ├── lib/
│   │   ├── Dialect/               # COADialect.cpp, COAOps.cpp
│   │   ├── Transforms/            # ShapeInfer / OpFusion / Tiling /
│   │   │                          # AddrAssign / Legalize
│   │   └── CodeGen/               # VLIWCodeGen.cpp（位域打包输出）
│   ├── tools/
│   │   ├── coa-opt/               # 调试工具：逐 Pass 运行与检查
│   │   └── coa-compiler/          # 一键编译：MLIR → VLIW 二进制
│   └── CMakeLists.txt
│
├── coa/                           # Python 参考库
│   ├── onnx_importer.py           # ONNX → COA MLIR 转换器
│   ├── mlir_parser.py             # COA MLIR → VLIW 参数解析器
│   ├── vliw.py                    # Python VLIW 位打包参考实现
│   └── tiling.py                  # Python Tiling 约束计算器
│
├── optimizer/                     # AI 优化模块
│   ├── tiling_search/
│   │   ├── env.py                 # OpenAI-Gym 兼容 Tiling 环境
│   │   ├── bayesian_opt.py        # 贝叶斯优化（Optuna TPE 采样）
│   │   └── rl_agent.py            # DQN 强化学习智能体
│   ├── op_fusion/
│   │   ├── graph_builder.py       # COA MLIR → PyTorch Geometric 图
│   │   ├── gnn_model.py           # FusionGAT 边分类模型
│   │   └── train_fusion.py        # GNN 训练脚本
│   └── auto_quant/
│       ├── sensitivity.py         # Fisher 信息量化敏感度分析
│       ├── pareto_search.py       # NSGA-II 三目标 Pareto 搜索
│       └── mixed_quant.py         # 混合精度 MLIR 属性重写
│
├── examples/
│   └── resnet18/                  # ResNet-18 端到端示例
│       ├── export_onnx.py         # PyTorch PTQ 量化导出
│       ├── import_mlir.py         # ONNX → COA MLIR
│       ├── compile.sh             # 调用 coa-compiler 编译
│       ├── verify.py              # VLIW 二进制逐字节验证
│       ├── model/resnet18.mlir    # 生成的 COA MLIR
│       └── output/resnet18.vliw  # 编译产物
│
├── tests/                         # 单元测试（26 tests）
│   ├── test_vliw_codegen.py       # VLIW 位打包 + Tiling 约束
│   └── test_ai_optimizer.py       # AI 优化模块测试
│
├── docs/                          # 项目文档（本目录）
├── build.sh                       # 顶层构建脚本
└── pyproject.toml                 # Python 包配置
```

---

## 4. 各程序功能详解

### 4.1 Python 前端库（`coa/`）

#### `coa/onnx_importer.py` — ONNX 模型导入器

**功能**：将量化 INT8 ONNX 模型转换为 COA MLIR（Level-1）文本文件。

**工作流程**：
1. 用 `onnx.load()` 解析模型，提取计算图节点和初始化权重
2. 构建张量形状字典（用于后续类型推断）
3. 遍历 ONNX 计算图，对每个支持的节点调用对应的 `_parse_xxx()` 函数
4. 收集所有未定义的 SSA 值（权重、偏置初始化张量），添加到函数参数列表
5. 通过 `_emit_mlir()` 拼接生成合法的 MLIR 文本

**支持的 ONNX 算子**：

| ONNX 算子 | COA MLIR 算子 | 备注 |
|---|---|---|
| `QLinearConv` | `coa.qlinearconv` | 解析 per-channel 量化参数 |
| `MaxPool` | `coa.maxpool` | 解析 kernel/stride/pad |
| `GlobalAveragePool` + `QuantizeLinear` | `coa.qlinearglobalaveragepool` | |
| `QLinearAdd` | `coa.qlinearadd` | 双输入残差加法 |
| `QGemm` | `coa.qgemm` | 全连接层，使用 `a_scale/b_scale` 命名 |

**属性类型映射**（MLIR 15 兼容）：
- scale 属性 → `f64`（对应 `F64Attr`/`F64ArrayAttr`）
- zp 数组 → `[v : i64]`（对应 `I64ArrayAttr`）
- kernel/stride/pad → `[v : i64]`（对应 `I64ArrayAttr`）
- 函数返回类型 → `tensor<?xi8>`（动态 shape，由 shape-infer 填充）

---

#### `coa/mlir_parser.py` — COA MLIR 解析器

**功能**：将**已完全降级**的 COA MLIR 文件（包含 tiling、地址等硬件属性）解析为 Python 字典列表，每个字典直接对应 `VLIW(**layer)` 的构造参数。

**工作流程**：
1. 用正则表达式匹配所有 `"coa.<opname>"(...) { ... }` 块
2. 对每个块提取整数属性（`R`, `C`, `M`, `N`, `tR`, `tC`, `tM`, `in_addr` 等）
3. 将 op 名映射到 VLIW operator 编码（Conv=0, Pool=1, GAP=2, Add=3）
4. 返回 `List[Dict]`，可直接传给 Python `VLIW` 类或用于验证

**注意**：输入 MLIR 必须经过完整流水线降级（Level-3），否则地址和 tiling 字段为 0。

---

#### `coa/vliw.py` — Python VLIW 位打包参考实现

**功能**：Python 版 VLIW 指令序列化，是 C++ `VLIW::toBytes()` 的参考实现，用于单元测试和验证对比。

**字段布局**：按 `VLIWDefs.h` 定义的 512-bit LSB-first 位域顺序打包，共 64 字节（见 §7 VLIW 指令结构）。

**使用场景**：
- `verify.py` 用它从 MLIR 重新生成参考 VLIW 二进制，与 `coa-compiler` 输出对比
- 单元测试 `test_vliw_codegen.py` 验证位打包逻辑

---

#### `coa/tiling.py` — Tiling 约束计算器

**功能**：Python 版硬件缓冲区约束检查与分块搜索，是 C++ `Tiling.cpp` 中 `checkBuffers()` 和 `getTile()` 的精确 Python 移植。

**三个关键约束**（与 C++ 完全对应）：

```
wdepth = tN32_max(N,tN) × k × k × tM16_max(M,tM)   < WDEPTH_LIMIT (256)
gdepth = ((tR-1)×s+(k-1)×d+1) × ((tC-1)×s+(k-1)×d+1) × tN32_max   < 1024
odepth = tR × tC × tM16_max(M,tM)                                   < 2048
```

其中 `tN32_max` 和 `tM16_max` 计算最坏情况下跨越的 32/16-wide bank 数，是 FPGA 存储体切分的关键参数。

**`get_tile(N, M, R, C, k, s, pad, d)`**：贪心分块搜索，从 `(M, R, C)` 开始逐步缩减，直到满足三项约束，再尝试扩大 `tR`。

---

### 4.2 C++ MLIR 编译器工具（`compiler/`）

#### `coa-compiler` — 一键端到端编译器

**功能**：完整编译流水线驱动器，输入 COA MLIR 文件，输出 VLIW 二进制（`.vliw`）。

**命令行参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `<input.mlir>` | （必填） | 输入 COA MLIR 文件路径 |
| `--output <file>` | `output.vliw` | 输出 VLIW 二进制文件路径 |
| `--weight-base <hex>` | `0x08000000` | 权重张量 DDR 基地址 |
| `--bias-base <hex>` | `0xC0000000` | 偏置张量 DDR 基地址 |
| `--act-base <hex>` | `0x10000000` | 激活张量 DDR 基地址（ping-pong） |
| `--wdepth-limit <n>` | `256` | 权重缓冲深度上限 |
| `--gdepth-limit <n>` | `1024` | 输入 tile 缓冲深度上限 |
| `--odepth-limit <n>` | `2048` | 输出 tile 缓冲深度上限 |
| `--skip-legalize` | `false` | 跳过 legalize 验证（调试用） |

**内部执行顺序**：
1. 解析命令行，建立 `MLIRContext` 并注册 COA 方言
2. 用 `parseSourceFile<ModuleOp>()` 解析输入 MLIR
3. 构建 `PassManager`，在 `func::FuncOp` 上嵌套 6 个 Pass
4. 调用 `pm.run(*module)` 执行流水线
5. `coa-vliw-gen` pass 直接将二进制写入指定输出文件

```bash
# 典型用法
coa-compiler --output resnet18.vliw \
             --weight-base 0x08000000 \
             --bias-base   0xC0000000 \
             --act-base    0x10000000 \
             model/resnet18.mlir
```

---

#### `coa-opt` — 逐 Pass 调试工具

**功能**：`mlir-opt` 风格的 MLIR 优化器驱动器，注册所有 COA Pass，支持单独运行任意 Pass 组合，用于开发和调试。

**典型用法**：

```bash
# 只运行 shape-infer，输出检查形状是否正确
coa-opt --coa-shape-infer model.mlir -o model_shape.mlir

# 跑到 tiling，检查 tM/tR/tC 分配
coa-opt --coa-shape-infer --coa-op-fusion --coa-tiling \
        model.mlir -o model_tiled.mlir

# 完整流水线，输出供 Python verify.py 对比的 lowered MLIR
coa-opt --coa-shape-infer --coa-op-fusion \
        --coa-tiling --coa-addr-assign --coa-legalize \
        model.mlir -o model_lowered.mlir

# 完整编译到 VLIW（带代码生成）
coa-opt --coa-shape-infer --coa-op-fusion \
        --coa-tiling --coa-addr-assign --coa-legalize \
        "--coa-vliw-gen=output=model.vliw" \
        model.mlir
```

与 `coa-compiler` 的区别：`coa-opt` 可以停在任意中间阶段检查 MLIR，而 `coa-compiler` 总是运行完整流水线并写出二进制。

---

### 4.3 MLIR 编译 Pass（`compiler/lib/`）

#### Pass 1 · `ShapeInfer.cpp` — 形状推断

**输入**：Level-1 COA MLIR（`R/C/M/N` 均为 0）
**输出**：Level-2 COA MLIR（张量维度属性已填入）

**核心逻辑**：
- 对 `qlinearconv`：计算 `R = floor((R0 + 2×pad - (k-1)×d - 1) / s + 1)`
- 对 `maxpool`：同上公式
- 对 `qlinearadd`：直接从输入 shape 复制
- 对 `qgemm`：从 weight shape 提取 M（输出通道数）
- 填写 `sM_concat`（Concat 通道偏移）和 `M_concat`（总通道数）

**实现技术**：遍历 `func::FuncOp` 内所有 op，通过 `op.getType().dyn_cast<RankedTensorType>()` 提取 shape，再用 `rewriter.updateRootInPlace()` 原地修改属性。

---

#### Pass 2 · `OpFusion.cpp` — 算子融合

**输入**：Level-2 COA MLIR
**输出**：Conv 算子的 `silu_addr` 属性被设为后继 Add 算子的输出地址（占位符）

**融合模式**：
```
%conv = coa.qlinearconv(...)
%add  = coa.qlinearadd(%conv, %shortcut, ...)
──────────────────────────────────────────────
↓ 融合后
%conv = coa.qlinearconv(..., silu_addr=<add_out_addr>, ...)
（Add 算子在 VLIW 层面被编码进 Conv 指令的 silu 字段）
```

**实现技术**：`OpRewritePattern<QLinearConvOp>`，通过 `value.getUsers()` 查找直接后继的 `QLinearAddOp`，匹配成功后用 `rewriter.updateRootInPlace` 修改 Conv 的属性，并调用 `applyPatternsAndFoldGreedily` 应用模式。

---

#### Pass 3 · `Tiling.cpp` — 硬件感知分块

**输入**：Level-2 COA MLIR
**输出**：每个算子填入 `tM`, `tR`, `tC` 属性

**核心函数 `getTile(N, M, R, C, k, s, pad, d)`**（与 `coa/tiling.py` 完全对应）：
1. 初始化 `(tM, tR, tC) = (M, R, C)`（最大 tile）
2. 检查三项缓冲区约束：
   - `wdepth < wdepth_limit`（默认 256）
   - `gdepth < gdepth_limit`（默认 1024）
   - `odepth < odepth_limit`（默认 2048）
3. 约束违反时按优先级缩减：`wdepth` 违反缩 `tM`；`gdepth`/`odepth` 违反先缩 `tR` 再缩 `tC`
4. 缩减完后尝试逐倍扩大 `tR` 以提高利用率

**Pass 选项**（可通过 `coa-compiler` 命令行配置）：
- `--wdepth-limit`（默认 256）
- `--gdepth-limit`（默认 1024）
- `--odepth-limit`（默认 2048）

---

#### Pass 4 · `AddrAssign.cpp` — DDR 地址分配

**输入**：Level-2+Tiling COA MLIR
**输出**：每个算子填入 `in_addr`, `out_addr`, `weight_addr`, `bias_addr`, `factor`, `factor2`

**地址分配策略**：

```
权重地址（顺序追加，不重叠）：
  weight_base = 0x08000000（可配置）
  每层 weight_offset += M × align32(N) × kH × kW  （字节）

偏置地址（顺序追加）：
  bias_base = 0xC0000000（可配置）
  每层 bias_offset += M × 4                         （int32 字节）

激活地址（Ping-Pong 双缓冲）：
  第 0 层输入 → 0x00000000（函数参数）
  其余层：in_addr ← 上一层 out_addr
  out_addr 在两个基地址间交替
  act_base = 0x10000000（可配置）
```

**量化因子计算**：
```
Conv/GEMM: factor  = round(in_scale × weight_scale[0] / out_scale × 2^15)
Add:       factor  = round(a_scale / out_scale × 2^15)
           factor2 = round(b_scale / out_scale × 2^15)
```

---

#### Pass 5 · `Legalize.cpp` — 合法性验证

**功能**：在进入代码生成之前，对所有算子的 VLIW 位域做边界检查，发现问题立即中止并报错。

**检查项**：

| 字段 | 位宽 | 最大值 | 检查逻辑 |
|---|---|---|---|
| `R`, `C`, `R0`, `C0` | 11 bit | 2047 | 尺寸不超限 |
| `M`, `N`, `tM`, `tN` | 12 bit | 4095 | 通道数不超限 |
| `tR`, `tC` | 11 bit | 2047 | tile 尺寸不超限 |
| `in_addr`, `out_addr`, `weight_addr` | 36 bit | 68GB | DDR 地址范围 |
| `factor` | 34 bit 有符号 | ±2^33 | 量化系数范围 |
| `factor2` | 32 bit 有符号 | ±2^31 | Add 量化系数范围 |

---

#### Pass 6 · `VLIWCodeGen.cpp` — VLIW 代码生成

**功能**：遍历所有 COA 算子，将 Level-3 属性打包为 64 字节 VLIW 指令，写出二进制文件。

**工作流程**：
1. 遍历 `func::FuncOp` 内每个 op，按 op 类型调用对应的 `fillVLIW_xxx()` 函数
2. 每个函数读取算子属性，填写 `VLIW` 结构体各字段
3. 调用 `VLIW::toBytes()` 将结构体序列化为 64 字节（LSB-first 位打包）
4. 所有指令字节追加写入输出文件

**Pass 选项**：`--output=<file>` 指定输出文件路径（默认 `output.vliw`）

---

### 4.4 示例程序（`examples/resnet18/`）

#### `export_onnx.py` — PyTorch PTQ 量化导出

**功能**：加载 PyTorch ResNet-18 浮点模型，进行训练后量化（PTQ），导出 INT8 ONNX 模型。

**输出**：`data/resnet18_quant_int8.onnx`

---

#### `import_mlir.py` — ONNX 转 COA MLIR

**功能**：调用 `coa.onnx_importer.convert_model()` 将量化 ONNX 转为 COA MLIR。

**输出**：`model/resnet18.mlir`（326 行，20+ 个算子）

---

#### `compile.sh` — 编译脚本

**功能**：调用 `coa-compiler` 运行完整流水线，并可选调用 `coa-opt` 导出 lowered MLIR 供检查。

```bash
# 脚本自动查找 compiler/build/bin/coa-compiler（相对路径）
bash examples/resnet18/compile.sh
```

**注意**：脚本中 `COA_COMPILER` 路径指向 `../../build/bin/coa-compiler`（相对于 `examples/resnet18/`），即项目根目录的 `build/bin/`，**实际编译产物在 `compiler/build/bin/`**，直接执行时需手动指定路径：

```bash
compiler/build/bin/coa-compiler \
    --output examples/resnet18/output/resnet18.vliw \
    --weight-base 0x08000000 --bias-base 0xC0000000 --act-base 0x10000000 \
    examples/resnet18/model/resnet18.mlir
```

---

#### `verify.py` — VLIW 二进制验证

**功能**：将 `coa-compiler` 生成的 `.vliw` 文件与 Python 参考实现（`coa.mlir_parser` + `coa.vliw`）的输出逐字节对比，生成详细差异报告。

**验证逻辑**：
1. 加载 `output/resnet18.vliw`，按 64 字节切分
2. 用 `mlir_parser.parse_all_layers()` 从 MLIR 提取参数
3. 用 Python `VLIW(**layer).to_bytes()` 构造参考字节
4. 逐指令逐字节对比，输出 `verify_report.txt`

**当前状态**：
- op 类型码全部匹配（拓扑结构正确）
- 量化零点字段存在偏差（`0x80` vs `0x00`），原因是 Python `mlir_parser` 对 `in_zp`/`out_zp` 字段解析默认值与 C++ 不一致，属 Python 参考实现问题而非编译器 bug

---

### 4.5 AI 优化模块（`optimizer/`）

#### `tiling_search/env.py` — RL Tiling 环境

**功能**：OpenAI Gym 兼容的 Tiling 搜索环境。状态空间为 13 维（层参数归一化 + 当前缓冲区利用率），动作空间为 9 种（tM/tR/tC 的 ×0.5 / ×2 / 提交），奖励为缓冲区利用率均值（违约 -1）。

#### `tiling_search/bayesian_opt.py` — 贝叶斯 Tiling 优化

**功能**：用 Optuna TPE 采样器在离散 `(tM, tR, tC)` 空间中最大化缓冲区利用率，输出 `tile_lookup.json` 查找表，可直接替换 `coa-tiling` Pass 的贪心结果。

#### `tiling_search/rl_agent.py` — DQN Tiling 智能体

**功能**：经验回放 + target network DQN，在随机生成的层参数分布上训练 5000 步，学会跨层泛化的 Tiling 策略。与贪心基线对比验证效果。

#### `op_fusion/graph_builder.py` — 计算图构建

**功能**：将 COA MLIR 文件解析为 PyTorch Geometric `Data` 对象。节点：每个 COA op（15 维特征 = 5 维 one-hot 类型 + 10 维标量），边：数据流方向（producer → consumer）。

#### `op_fusion/gnn_model.py` — FusionGAT 融合预测

**功能**：图注意力网络边分类器，输出每条数据流边的融合概率。架构：`GATConv(15→64, heads=4) → ELU → GATConv(256→64) → 边 MLP → sigmoid`。已知可融合模式：`qlinearconv→qlinearadd`（残差）、`qlinearconv→maxpool`（Conv-Pool）。

#### `auto_quant/sensitivity.py` — 量化敏感度分析

**功能**：基于 Fisher 信息对角近似（HAWQ 风格），计算每层 4/8 bit 量化的误差代理 `||W - Q(W,bits)||²_F`，对层按敏感度排序。

#### `auto_quant/pareto_search.py` — Pareto 混合精度搜索

**功能**：NSGA-II 三目标优化（精度损失代理 / 硬件开销 / 缓冲峰值），输出 Pareto 前沿 JSON 和最优混合精度配置。

#### `auto_quant/mixed_quant.py` — 混合精度 MLIR 重写

**功能**：读取 `mixed_precision_config.json`，将对应层的量化 scale 属性重写为新精度对应的值。

---

## 5. COA 方言（MLIR 算子定义）

COA 方言（Coarse-grained Operator Accelerator）是本项目定义的自定义 MLIR 方言，用 TableGen ODS 描述，C++ 命名空间为 `mlir::coa`，方言前缀为 `coa`。

### 支持的算子

| 算子 | VLIW op 编码 | 说明 |
|---|---|---|
| `coa.qlinearconv` | 0 (Conv) | INT8 量化 2D 卷积，支持 per-channel 权重量化 |
| `coa.qgemm` | 0 (Conv) | INT8 量化全连接（GEMM），映射为 1x1 卷积 |
| `coa.maxpool` | 1 (Pool) | 整数 2D 最大池化 |
| `coa.qlinearglobalaveragepool` | 2 (GAP) | 量化全局平均池化 |
| `coa.qlinearadd` | 3 (Add) | 量化元素加法（残差连接） |

### 属性层级（三层渐进式填充）

每个算子的属性由各 Pass 逐层填入，一份 MLIR 文件会在流水线中不断被丰富：

```
Level 1  —— ONNX 层（import_mlir.py 填入，必须）
  in_scale, in_zp             输入激活量化参数（f64, i32）
  weight_scale, weight_zp     权重量化参数（f64 数组, i64 数组）
  out_scale, out_zp           输出激活量化参数（f64, i32）
  kernel_shape, strides       卷积核尺寸与步长（i64 数组）
  pads, dilations             填充与膨胀率（i64 数组，可选）

Level 2  —— 形状层（coa-shape-infer 填入）
  R, C      输出特征图高度/宽度（11-bit，max 2047）
  M, N      输出/输入通道数（12-bit，max 4095）
  R0, C0    输入特征图高度/宽度
  sM_concat, M_concat   Concat 通道元数据

Level 3  —— 硬件层（coa-tiling + coa-addr-assign 填入）
  tM, tR, tC            分块大小（缓冲区感知）
  in_addr, out_addr     激活 DDR 地址（36-bit）
  weight_addr           权重 DDR 地址（36-bit）
  bias_addr             偏置 DDR 地址（11-bit 页偏移）
  silu_addr             残差融合输出地址（coa-op-fusion 设置）
  factor, factor2       定点量化系数（34-bit/32-bit 有符号）
```

### TableGen 定义文件

| 文件 | 内容 |
|---|---|
| `compiler/include/COA/COADialect.td` | 方言声明，方言名 `coa`，namespace `mlir::coa` |
| `compiler/include/COA/COAOps.td` | 5 个算子的完整 ODS 定义（参数、约束、属性） |
| `compiler/include/COA/COAPasses.td` | 6 个 Pass 的 TableGen 定义（名称、选项、描述） |

---

## 6. 编译器 Pass 流水线

### Pass 依赖与执行顺序

```
输入：Level-1 COA MLIR（ONNX 量化参数 + 卷积几何）
  │
  ▼  Pass 1: coa-shape-infer
  │   推断每个算子的 R,C,M,N,R0,C0（输出/输入空间维度）
  │   公式：R = floor((R0 + 2*pad - (k-1)*d - 1) / s + 1)
  │
  ▼  Pass 2: coa-op-fusion
  │   识别 Conv → Add 残差连接模式
  │   将 Conv 的 silu_addr 设为后继 Add 的输出地址（预留位）
  │
  ▼  Pass 3: coa-tiling  [可配置深度限制]
  │   贪心求解最大合法分块 (tM, tR, tC)
  │   约束：wdepth < 256, gdepth < 1024, odepth < 2048
  │
  ▼  Pass 4: coa-addr-assign  [可配置 DDR 基地址]
  │   权重/偏置：顺序线性追加
  │   激活：Ping-Pong 双缓冲（交替两个地址池）
  │   计算量化因子：factor = round(s_in * s_w / s_out * 2^15)
  │
  ▼  Pass 5: coa-legalize
  │   验证所有 VLIW 位域不超出硬件约束
  │   发现溢出立即 signalPassFailure() 并打印诊断信息
  │
  ▼  Pass 6: coa-vliw-gen  [可配置输出文件]
     将每个算子打包为 64 字节 VLIW 指令
     写出 N * 64 字节二进制文件

输出：Level-3 COA MLIR（所有属性填满） + .vliw 二进制
```

### Pass 选项完整列表

```bash
# coa-tiling 选项
--coa-tiling="wdepth-limit=256 gdepth-limit=1024 odepth-limit=2048"

# coa-addr-assign 选项
--coa-addr-assign="weight-base=134217728 bias-base=3221225472 act-base=268435456"
# 十六进制等价：weight-base=0x08000000, bias-base=0xC0000000, act-base=0x10000000

# coa-vliw-gen 选项
--coa-vliw-gen="output=model.vliw"
```

---

## 7. VLIW 指令结构

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

## 8. AI 优化模块

### Phase 3-A：Tiling 自动搜索（`optimizer/tiling_search/`）

#### 贝叶斯优化（`bayesian_opt.py`）

使用 **Optuna TPE 采样器**在离散空间 `(tM, tR, tC)` 中最大化缓冲区利用率：

```
score = (wdepth/wLimit + gdepth/gLimit + odepth/oLimit) / 3
约束：所有利用率 <= 1.0（不超限）
```

输出 `tile_lookup.json`，可替换 `coa-tiling` Pass 的贪心搜索。

```python
from optimizer.tiling_search.bayesian_opt import bayesian_tile_search
tM, tR, tC = bayesian_tile_search(N=64, M=256, R=56, C=56, k=3, s=1, d=1)
```

#### DQN 强化学习（`rl_agent.py`）

- **状态**（13 维）：层参数归一化 + 当前 tile 缓冲区利用率
- **动作**（9 种）：tM/tR/tC 各自 ×0.5 / ×2 / 提交
- **奖励**：合法时返回三项利用率均值，违约返回 -1
- **训练**：随机生成层参数分布，5000 步 DQN 训练

```bash
python optimizer/tiling_search/rl_agent.py  # 训练并与贪心基线对比
```

---

### Phase 3-B：GNN 算子融合（`optimizer/op_fusion/`）

#### 图构建（`graph_builder.py`）

将 COA MLIR 解析为 PyTorch Geometric `Data` 对象：
- **节点**：每个 COA op，15 维特征（5 维 one-hot 类型 + 10 维标量属性）
- **边**：数据流方向（producer → consumer）

#### FusionGAT 模型（`gnn_model.py`）

```
输入:  节点特征 [N, 15]
       GATConv(15 → 64, heads=4) → ELU
       GATConv(256 → 64, heads=1) → ELU
       边 MLP: cat(h_src, h_dst) → Linear(128→1) → sigmoid
输出:  每条边的融合概率（0~1）
```

已知可融合模式（规则标注训练数据）：

| 模式 | 含义 |
|---|---|
| `qlinearconv → qlinearadd` | Conv + 残差加法，融合为单 VLIW 指令 |
| `qlinearconv → maxpool` | Conv-Pool 链，减少中间激活写回 |
| `qlinearconv → qlinearglobalaveragepool` | 末层 Conv-GAP 融合 |

```bash
python optimizer/op_fusion/train_fusion.py --mlir-dir examples/ --epochs 100
```

---

### Phase 3-C：AutoQuant 混合精度量化（`optimizer/auto_quant/`）

#### 敏感度分析（`sensitivity.py`）

Fisher 信息对角近似（HAWQ 风格），计算每层 4/8-bit 量化误差代理：
```
sensitivity[bits] = ||W - Q(W, bits)||^2_F
```
层按 4-bit 误差从高到低排列，敏感度高的层分配更多比特。

#### NSGA-II Pareto 搜索（`pareto_search.py`）

三目标同时最小化：

| 目标 | 计算 | 含义 |
|---|---|---|
| f1 精度损失代理 | Fisher 误差加权和 | 量化对精度的影响 |
| f2 硬件开销 | 均值比特宽度 / 16 | 指令/内存开销 |
| f3 缓冲峰值压力 | 最大层权重字节 / 32MB | 片上内存压力 |

```bash
python optimizer/auto_quant/pareto_search.py model_weights.npz 100 50
# 输出：pareto_front.json, mixed_precision_config.json
```

#### 应用混合精度（`mixed_quant.py`）

```bash
python optimizer/auto_quant/mixed_quant.py \
    --config mixed_precision_config.json \
    --mlir   model.mlir \
    --output model_mixed.mlir
```

---

## 9. 测试体系

```bash
# 从项目根目录运行全部 Python 测试
python -m pytest tests/ -v
```

**当前状态：26/26 全部通过**

| 测试类 | 文件 | 覆盖内容 |
|---|---|---|
| `TestPythonVLIWPacking` | `test_vliw_codegen.py` | VLIW 位域打包正确性（8 tests） |
| `TestExtractVLIW` | `test_vliw_codegen.py` | MLIR → 参数解析流程（1 test） |
| `TestTilingConstraints` | `test_vliw_codegen.py` | 缓冲区约束检查（3 tests） |
| `TestBayesianTiling` | `test_vliw_codegen.py` | 贝叶斯搜索合法性（1 test） |
| `TestTilingEnv` | `test_ai_optimizer.py` | RL 环境状态/动作/奖励（5 tests） |
| `TestGraphBuilder` | `test_ai_optimizer.py` | GNN 图节点/边构建（3 tests） |
| `TestSensitivity` | `test_ai_optimizer.py` | 量化敏感度计算（3 tests） |
| `TestMixedQuant` | `test_ai_optimizer.py` | 混合精度 MLIR 重写（2 tests） |

---

## 10. 构建说明

### 环境要求

| 依赖 | 版本 | 用途 |
|---|---|---|
| LLVM/MLIR | **15**（系统安装） | C++ 编译器依赖 |
| CMake | >= 3.20 | 构建系统 |
| Python | >= 3.9 | onnx_importer / optimizer |
| `onnx` | >= 1.14 | ONNX 模型解析 |
| `numpy` | >= 1.24 | 权重处理 |

### Linux 快速构建（已验证）

系统已安装 `llvm-15-dev` 和 `mlir-15`（`/usr/lib/llvm-15`）：

```bash
# 构建 C++ 编译器（约 2-5 分钟）
cmake -S compiler -B compiler/build \
      -DMLIR_DIR=/usr/lib/llvm-15/lib/cmake/mlir \
      -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm \
      -DCMAKE_BUILD_TYPE=Release
cmake --build compiler/build -j$(nproc)
```

构建产物：
- `compiler/build/bin/coa-compiler` — 端到端编译器
- `compiler/build/bin/coa-opt` — 逐 Pass 调试工具

### Python 依赖安装

```bash
# 核心前端依赖
pip install onnx numpy

# AI 优化模块（按需安装）
pip install optuna          # Phase 3-A 贝叶斯优化
pip install torch           # Phase 3-A DQN / Phase 3-B GNN
pip install torch-geometric # Phase 3-B FusionGAT
pip install pymoo           # Phase 3-C NSGA-II
```

---

## 11. 完整使用示例（ResNet-18）

### 步骤 1：导出量化 ONNX（已有则跳过）

```bash
python examples/resnet18/export_onnx.py
# 输出：examples/resnet18/data/resnet18_quant_int8.onnx
```

### 步骤 2：ONNX → COA MLIR

```bash
python -m examples.resnet18.import_mlir
# 输出：examples/resnet18/model/resnet18.mlir（326 行，30 个算子）
```

生成的 MLIR 片段示例（Level-1，含量化参数）：
```mlir
module {
  func.func @resnet18(%input: tensor<1x3x224x224xi8>, %conv1_weight: tensor<...>,
                      %conv1_bias: tensor<...>, ...) -> tensor<?xi8> {
    %conv1 = "coa.qlinearconv"(%input, %conv1_weight, %conv1_bias) {
      in_scale = 0.00781250 : f64,
      in_zp = 128 : i32,
      weight_scale = [0.000755 : f64],
      weight_zp = [0 : i64],
      out_scale = 0.00390625 : f64,
      out_zp = 0 : i32,
      kernel_shape = [7 : i64, 7 : i64],
      strides = [2 : i64, 2 : i64],
      pads = [3 : i64, 3 : i64, 3 : i64, 3 : i64]
    } : (tensor<?xi8>, tensor<?xi8>, tensor<?xi32>) -> tensor<?xi8>
    ...
  }
}
```

### 步骤 3：编译到 VLIW 二进制

```bash
compiler/build/bin/coa-compiler \
    --output examples/resnet18/output/resnet18.vliw \
    --weight-base 0x08000000 \
    --bias-base   0xC0000000 \
    --act-base    0x10000000 \
    examples/resnet18/model/resnet18.mlir

# 预期输出：
# [coa-vliw-gen] Wrote 30 VLIW instructions (1920 bytes) to resnet18.vliw
# coa-compiler: Done. Output written to resnet18.vliw
```

### 步骤 4：验证输出

```bash
python -m examples.resnet18.verify
# 输出：examples/resnet18/output/verify_report.txt
# 30/30 op 类型码匹配；零点字段偏差为 Python 参考问题，非编译器 bug
```

### 逐 Pass 调试（使用 coa-opt）

```bash
BIN=compiler/build/bin/coa-opt
MLIR=examples/resnet18/model/resnet18.mlir

# 只做形状推断，检查 R/C/M/N 是否正确
$BIN --coa-shape-infer $MLIR -o /tmp/shape.mlir

# 到 tiling，检查 tM/tR/tC 分配
$BIN --coa-shape-infer --coa-op-fusion --coa-tiling $MLIR -o /tmp/tiled.mlir

# 完整流水线，导出 lowered MLIR（可供 Python verify 对比）
$BIN --coa-shape-infer --coa-op-fusion \
     "--coa-tiling=wdepth-limit=256 gdepth-limit=1024 odepth-limit=2048" \
     "--coa-addr-assign=weight-base=134217728 bias-base=3221225472 act-base=268435456" \
     --coa-legalize \
     $MLIR -o /tmp/lowered.mlir
```

---

## 12. 路线图

| 阶段 | 状态 | 内容 |
|---|---|---|
| Phase 1-A | ✅ 完成 | COA Dialect TableGen ODS + CMake 构建系统 |
| Phase 1-B | ✅ 完成 | 6 个 MLIR Pass C++ 实现（ShapeInfer/OpFusion/Tiling/AddrAssign/Legalize/VLIWGen） |
| Phase 1-C/D | ✅ 完成 | VLIW 代码生成后端 + `coa-opt` / `coa-compiler` 工具 |
| Phase 2 | ✅ 完成 | 测试体系（26 tests passing） |
| Phase 3-A | ✅ 完成 | Tiling 自动搜索（贝叶斯 TPE + DQN） |
| Phase 3-B | ✅ 完成 | GNN 算子融合（FusionGAT 边分类器） |
| Phase 3-C | ✅ 完成 | AutoQuant 混合精度量化（NSGA-II Pareto） |
| **M8** | ✅ 完成 | LLVM-15 系统安装 + `onnx_importer` 修复 + ResNet-18 端到端验证（30 条 VLIW） |
| M9 | 🔲 计划中 | `verify.py` Python 参考实现对齐（量化零点字段修正） |
| M10 | 🔲 计划中 | YOLO / MobileNet 等更多模型端到端验证 |
| M11 | 🔲 计划中 | 学术论文：AI-Empowered MLIR Compiler for FPGA VLIW |
