# FPGA-MLIR 项目总结文档

> 更新时间：2026-05-08

---

## 项目概述

FPGA-MLIR 是一个面向自定义 FPGA 粗粒度 VLIW 指令集的 AI 编译器。完整流水线从**浮点 ONNX 模型**出发，经过量化、参数提取、MLIR 转换，最终生成 **512-bit VLIW 二进制指令流**，并产出 FPGA 板载可直接加载的权重/偏置/激活 memory image 文件。

```
Float ONNX
    │
    ▼  [coa.quantize]  PTQ 量化（INT8 QOperator）
INT8 ONNX
    ├──▶  [coa.hw_export]   → weight.image / bias.image / act.image / factors.json
    └──▶  [coa.onnx_importer] → Level-1 COA MLIR
                                    │
                                    ▼  [coa-compiler C++]  Pass 流水线
                               VLIW Binary (.vliw)
```

---

## 目录结构

```
FPGA-MLIR/
├── coa/                    # Python 前端库（核心算法）
│   ├── quantize.py         # PTQ 量化工具包
│   ├── hw_export.py        # 硬件参数提取
│   ├── onnx_importer.py    # 量化 ONNX → Level-1 COA MLIR
│   ├── mlir_parser.py      # COA MLIR → 层参数字典
│   ├── vliw.py             # 512-bit VLIW 指令封装
│   ├── tiling.py           # Buffer 约束检查 + 贪心 Tile 搜索
│   └── pruning.py          # 结构化剪枝
├── compiler/               # C++ MLIR 编译器（TableGen ODS + 6 Passes）
├── optimizer/              # AI 优化模块（贝叶斯 / DQN / GNN / NSGA-II）
├── examples/resnet18/      # 端到端 ResNet-18 示例
│   ├── model/resnet18.onnx       # 输入浮点模型
│   ├── export_onnx.py            # Step 1: 量化
│   ├── import_mlir.py            # Step 2: MLIR 转换
│   ├── compile.sh                # Step 3: VLIW 编译
│   ├── data/resnet18_quant_int8.onnx   # 量化后 ONNX（自动生成）
│   ├── model/resnet18.mlir             # COA MLIR（自动生成）
│   ├── output/resnet18.vliw            # VLIW 指令（自动生成）
│   └── parameters/                     # 硬件参数（自动生成）
│       ├── weight.image
│       ├── bias.image
│       ├── act.image
│       └── factors.json
├── tools/                  # 辅助工具脚本
├── tests/                  # 单元测试（26 tests passing）
└── legacy/                 # 遗留参考实现（用于对齐验证）
    ├── parameters/         # 参考硬件参数（legacy ground truth）
    └── datasets/calibration_data/  # 校准数据集（20 张 ImageNet 图像）
```

---

## 完整执行流程（以 ResNet-18 为例）

```bash
# Step 1: 量化（model/resnet18.onnx → data/resnet18_quant_int8.onnx）
python -m examples.resnet18.export_onnx

# Step 2: ONNX → COA MLIR
python -m examples.resnet18.import_mlir

# Step 3: MLIR → VLIW 二进制
compiler/build/bin/coa-compiler \
    --output examples/resnet18/output/resnet18.vliw \
    --weight-base 0x08000000 \
    --bias-base   0xC0000000 \
    --act-base    0x10000000 \
    examples/resnet18/model/resnet18.mlir

# Step 4: 提取硬件参数（weight/bias/act/factors）
python -c "
from coa.hw_export import export_hw_params
export_hw_params('examples/resnet18/data/resnet18_quant_int8.onnx',
                 'examples/resnet18/parameters/', force_per_tensor=True)
"
```

---

## 已实现功能

### 1. PTQ 量化工具包（`coa/quantize.py`）

#### 量化格式
| 格式 | 说明 |
|------|------|
| `int8`（默认） | 有符号非对称 INT8，zp ∈ [-128, 127]，与 onnxruntime QInt8 一致 |
| `int8-sym` | 有符号对称 INT8，zp = 0 |
| `uint8` | 无符号非对称 UINT8，zp ∈ [0, 255] |

#### 校准算法
- **MinMax**：使用观测到的最大/最小值
- **Percentile**：截断尾部百分位（默认 99.99%）
- **Entropy（KL 散度）**：最小化量化误差的熵

#### 权重量化
- **Per-Tensor**（默认，与 legacy 对齐）
- **Per-Channel**：逐输出通道量化

#### 支持的算子融合
- `Conv + ReLU` → `QLinearConv`（输出直接为 ReLU 后的 tensor）
- `Add + ReLU` → `QLinearAdd`
- `Gemm + ReLU` → `QGemm`

#### 支持的量化算子
| 浮点算子 | 量化后算子 |
|----------|------------|
| Conv | QLinearConv |
| Gemm | QGemm（com.microsoft） |
| Add（双激活输入） | QLinearAdd |
| MaxPool | 透传（不量化） |
| GlobalAveragePool | 透传（不量化） |

#### 高级优化
- **SmoothQuant**：激活-权重联合缩放（α 可调）
- **GPTQ**：Hessian 引导的权重误差修正

#### 硬件感知后处理（hw_aware passes）
- `hw_fix_scale_factors`：将 M = in_scale × w_scale / out_scale 限制在硬件有效范围内
- `hw_equalize_add_scales`：平衡 QLinearAdd 两个分支的重量化因子
- `hw_check_factors`：逐层打印因子合法性报告

#### 校准推理后端
- 使用 onnxruntime InferenceSession（精度高，与实际推理一致）
- 自动扩展中间 tensor 为模型输出，收集完整激活统计

---

### 2. 硬件参数提取（`coa/hw_export.py`）

#### 输出文件
| 文件 | 内容 | 格式 |
|------|------|------|
| `weight.image` | INT8 权重（FPGA layout：OC×KH×KW×IC_pad32） | 二进制 |
| `bias.image` | INT32 偏置（OC 对齐到 4 的倍数） | 二进制 |
| `act.image` | 256-entry INT8 激活 LUT（Sigmoid/Clip 等） | 二进制 |
| `factors.json` | 每层定点重量化因子及地址偏移 | JSON |
| `weight_offset.txt` | 每层权重在 weight.image 中的偏移 | 文本 |
| `bias_offset.txt` | 每层偏置在 bias.image 中的偏移 | 文本 |
| `act_offset.txt` | 每层激活 LUT 在 act.image 中的偏移 | 文本 |

#### 支持的节点类型
- `QLinearConv` → `ConvParams`（含权重、偏置、scale、zp）
- `QGemm` → `ConvParams`（同上，用于全连接层）
- `QLinearAdd` → `AddParams`（两路分支 scale/zp）
- 激活函数（ReLU/Clip/Sigmoid）→ `ActLUT`（256 entry 查找表）

#### Per-Channel → Per-Tensor 转换
- 自动将 per-channel 量化权重转换为 per-tensor（`force_per_tensor=True`）
- 转换步骤：反量化 → 重新计算全局 scale → 重新量化 + 调整 bias

---

### 3. ONNX → COA MLIR 转换（`coa/onnx_importer.py`）

#### 支持的 ONNX 算子
| ONNX 算子 | COA MLIR 算子 |
|-----------|---------------|
| QLinearConv | `coa.qlinearconv` |
| QGemm | `coa.qgemm` |
| QLinearAdd | `coa.qlinearadd` |
| MaxPool | `coa.maxpool` |
| GlobalAveragePool / QLinearAveragePool | `coa.qlinearglobalaveragepool` |

#### 特性
- 自动运行 ONNX shape inference（不严格检查 contrib op）
- Initializer shape 自动注入 shapes 字典
- QLinearAdd 输出 shape 从输入推导
- SSA 名称自动修正（数字开头自动加 `v` 前缀，MLIR 合法）
- 函数返回类型与实际 shape 一致（避免 coa-compiler 类型不匹配）

---

### 4. C++ 编译器流水线（`compiler/`）

#### 编译 Pass 顺序
1. **coa-shape-infer**：推断所有张量的静态形状
2. **coa-op-fusion**：算子融合（Conv+Add、Conv+ReLU 等）
3. **coa-tiling**：Tile 计算（满足 Buffer 容量约束）
4. **coa-addr-assign**：DDR/SRAM 地址分配（weight/bias/act）
5. **coa-legalize**：硬件合法性检查（维度、数据类型）
6. **coa-vliw-gen**：生成 512-bit VLIW 二进制

#### 内存地址空间（ResNet-18）
| 区域 | Base 地址 |
|------|-----------|
| Weight DDR | `0x08000000` |
| Bias DDR | `0xC0000000` |
| Activation SRAM | `0x10000000` |

---

### 5. Python Buffer / Tiling 工具（`coa/tiling.py`）

- `calculate_buffer_consumption(tile)` — 计算给定 tile 的 on-chip buffer 占用
- `get_tile(layer, budget)` — 贪心搜索满足 buffer 预算的最大合法 tile
- `buffer_utilization(tile)` — 打印各 buffer 区域利用率

---

### 6. 结构化剪枝（`coa/pruning.py`）

- `prune_onnx(model, sparsity, criterion)` — 按 L1/L2/随机 准则对 Conv 通道剪枝
- 输出仍为 ONNX 浮点模型，可直接进入量化流程

---

### 7. AI 优化模块（`optimizer/`）

- **贝叶斯 Tiling 优化**：基于 GP 代理模型搜索最优分块策略
- **DQN 调度器**：强化学习自动决策算子调度顺序
- **FusionGAT**：图注意力网络预测算子融合收益
- **NSGA-II AutoQuant**：多目标进化算法自动搜索量化配置（精度 vs 速度）

---

### 8. 辅助工具（`tools/`）

| 脚本 | 功能 |
|------|------|
| `compare_hw_params.py` | 逐字节对比两套 weight/bias/act.image，输出差异统计 |
| `quant_export_data.py` | 提取每层量化指标（SNR、余弦相似度、MAE、RMSE）并输出 JSON |
| `quant_plot.py` | 量化误差可视化（逐层分布图） |
| `quant_report.py` | 生成量化质量 HTML 报告 |

---

## 量化对齐验证（与 legacy 参考的差异）

| 文件 | 差异 | 说明 |
|------|------|------|
| `weight.image` | **0 / 11,769,856 字节（0%）** | 完全一致 ✅ |
| `bias.image` | **2 / 23,232 字节（0.01%）** | 丸め误差，可接受 ✅ |
| `factors.json` | 数值接近，部分层有微小差异 | 由激活 scale 差异引起 |

> **关键配置**：`weight_per_channel=False`（per-tensor），`calibration=minmax`，
> 使用 `legacy/datasets/calibration_data/` 下 20 张校准图像。

---

## 已知限制 / 待实现

- `verify.py` Python 参考对齐：zero_point=-128（signed asymmetric）在 Python VLIW 参考中尚未正确编码
- `onnx_importer` 不支持 `Flatten`、`GlobalAveragePool`（量化后）的 shape 传播
- `export_onnx.py` 的 `--weight-per-channel` 命令行参数默认已改为 per-tensor

---

*由 `coa.quantize` + `coa.hw_export` + `coa.onnx_importer` + `coa-compiler` 联合构成完整端到端链路。*
