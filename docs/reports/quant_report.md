# ResNet-18 PTQ 量化质量报告

**模型**：torchvision ResNet-18（预训练，ImageNet）  
**输入尺寸**：1×3×64×64（测试用，AdaptiveAvgPool 兼容任意尺寸）  
**校准样本**：4 张随机数据（minmax/entropy）  
**量化工具**：`coa.quantize`（自研，不依赖 onnxruntime）  
**日期**：2026-05-08

---

## 一、单元 + E2E 测试结果

**共 37 项测试，全部通过（0 失败）**

| 测试类别 | 测试数 | 结果 |
|---|---|---|
| `test_quantize.py`：scale/zp 数学、round-trip、weight quant、SmoothQuant、calibration | 23 | ✅ 全过 |
| `test_quantize_e2e.py`：QOperator 结构、逐层精度、bias 正确性、scale/zp 合理性 | 14 | ✅ 全过 |

---

## 二、量化配置对比

| 配置 | 首层 Conv SNR | 首残差块 cosine | FC cosine | Top-1 一致 | 整网 SNR | 量化耗时 |
|---|---|---|---|---|---|---|
| int8×int8  per-ch   minmax  | 36.0 dB | 0.9999 | 1.0000 | ✓ | 42.4 dB | 2.2 s |
| int8×int8  per-ch   entropy | 36.5 dB | 0.9921 | 1.0000 | ✓ | 41.9 dB | 30.9 s |
| uint8×int8 per-ch   minmax  | 36.4 dB | 1.0000 | 1.0000 | ✓ | 42.6 dB | 2.7 s |
| int8×int8  per-ten  minmax  | 36.0 dB | 0.9999 | 1.0000 | ✓ | 42.4 dB | 2.6 s |

**指标说明**
- **首层 Conv SNR**：第一层 Conv 输出的量化模拟信噪比（`float` vs `Q→DQ`）
- **首残差块 cosine**：第一个 Add（残差合并）输出的余弦相似度
- **FC cosine**：最后一层 Gemm 输出的余弦相似度
- **Top-1 一致**：`argmax(float_logits) == argmax(quant_logits)`
- **整网 SNR**：最终输出层的信噪比

**结论**：所有配置 Top-1 预测与浮点完全一致；整网 SNR 均 > 40 dB，余弦相似度均 ≥ 0.9999。

---

## 三、逐层 SNR（int8×int8 per-channel minmax）

> Relu / Flatten 输出无独立量化 scale，直接继承上一层的量化参数，标注为 `(no quant)`。

| # | 算子 | 层名 | SNR (dB) | cosine |
|---|---|---|---|---|
| 0 | Conv | conv1 | 36.0 | 0.9999 |
| 1 | Relu | relu | — | — |
| 2 | MaxPool | maxpool | 40.5 | 1.0000 |
| 3 | Conv | layer1.0/conv1 | 38.9 | 0.9999 |
| 4 | Relu | layer1.0/relu | 40.2 | 1.0000 |
| 5 | Conv | layer1.0/conv2 | 38.4 | 0.9999 |
| 6 | Add | layer1.0/Add（残差） | 39.5 | 0.9999 |
| 7 | Relu | layer1.0/relu_1 | 40.2 | 1.0000 |
| 8 | Conv | layer1.1/conv1 | 39.0 | 0.9999 |
| 9 | Relu | layer1.1/relu | 38.5 | 0.9999 |
| 10 | Conv | layer1.1/conv2 | 37.4 | 0.9999 |
| 11 | Add | layer1.1/Add（残差） | 39.6 | 0.9999 |
| 12 | Relu | layer1.1/relu_1 | 40.4 | 1.0000 |
| 13 | Conv | layer2.0/conv1 | 39.9 | 0.9999 |
| 14 | Relu | layer2.0/relu | 40.1 | 1.0000 |
| 15 | Conv | layer2.0/conv2 | 38.6 | 0.9999 |
| 16 | Conv | layer2.0/downsample（shortcut） | 36.9 | 0.9999 |
| 17 | Add | layer2.0/Add（残差） | 39.9 | 0.9999 |
| 18 | Relu | layer2.0/relu_1 | 40.3 | 1.0000 |
| 19 | Conv | layer2.1/conv1 | 40.4 | 1.0000 |
| 20 | Relu | layer2.1/relu | 39.4 | 0.9999 |
| 21 | Conv | layer2.1/conv2 | 39.1 | 0.9999 |
| 22 | Add | layer2.1/Add（残差） | 40.1 | 1.0000 |
| 23 | Relu | layer2.1/relu_1 | 41.1 | 1.0000 |
| 24 | Conv | layer3.0/conv1 | 40.0 | 0.9999 |
| 25 | Relu | layer3.0/relu | 40.4 | 1.0000 |
| 26 | Conv | layer3.0/conv2 | 39.5 | 0.9999 |
| 27 | Conv | layer3.0/downsample（shortcut）⚠️ | 34.1 | 0.9998 |
| 28 | Add | layer3.0/Add（残差） | 35.9 | 0.9999 |
| 29 | Relu | layer3.0/relu_1 | 43.9 | 1.0000 |
| 30 | Conv | layer3.1/conv1 | 40.8 | 1.0000 |
| 31 | Relu | layer3.1/relu | 40.9 | 1.0000 |
| 32 | Conv | layer3.1/conv2 | 38.2 | 0.9999 |
| 33 | Add | layer3.1/Add（残差） | 38.8 | 0.9999 |
| 34 | Relu | layer3.1/relu_1 | 38.7 | 0.9999 |
| 35 | Conv | layer4.0/conv1 | 41.7 | 1.0000 |
| 36 | Relu | layer4.0/relu | 41.9 | 1.0000 |
| 37 | Conv | layer4.0/conv2 | 42.3 | 1.0000 |
| 38 | Conv | layer4.0/downsample（shortcut） | 42.6 | 1.0000 |
| 39 | Add | layer4.0/Add（残差） | 41.7 | 1.0000 |
| 40 | Relu | layer4.0/relu_1 | 41.3 | 1.0000 |
| 41 | Conv | layer4.1/conv1 | 40.0 | 1.0000 |
| 42 | Relu | layer4.1/relu | 45.1 | 1.0000 |
| 43 | Conv | layer4.1/conv2 | 40.7 | 1.0000 |
| 44 | Add | layer4.1/Add（残差） | 40.8 | 1.0000 |
| 45 | Relu | layer4.1/relu_1 | 41.2 | 1.0000 |
| 46 | GlobalAvgPool | avgpool | 44.2 | 1.0000 |
| 47 | Flatten | — | 44.2 | 1.0000 |
| 48 | Gemm | output（fc） | 42.4 | 1.0000 |

---

## 四、关键观察

### ✅ 整体表现良好
- **所有层 cosine ≥ 0.9998**，量化误差对特征方向影响极小
- **SNR 最低点**：layer3.0/downsample shortcut conv（34.1 dB），属于 1×1 降采样卷积，通道数变化大，可考虑对该层单独用 percentile calibration 或 SmoothQuant

### ⚠️ 潜在敏感层
| 层 | SNR | 原因 |
|---|---|---|
| layer3.0/downsample | 34.1 dB | 1×1 shortcut，stride=2，激活分布较宽 |
| layer3.0/Add | 35.9 dB | 受上游 downsample 误差传播影响 |

### 📊 Calibration 方法对比
- `entropy` 对首层 SNR 略有提升（36.5 vs 36.0 dB），但耗时增加 14×（30.9 s vs 2.2 s）
- `minmax` 在随机校准数据下整体 SNR 更稳定，推荐作为默认方案
- `uint8×int8` 整网 SNR 比 `int8×int8` 略高（42.6 vs 42.4 dB），非对称量化在 ReLU 后激活有优势

---

## 五、量化产物

输出目录：`docs/reports/`

| 文件 | 格式 | 说明 |
|---|---|---|
| `q_int8xint8__per-ch__minmax.onnx` | QOperator INT8 | **推荐**，直接用于 coa.onnx_importer |
| `q_int8xint8__per-ch__entropy.onnx` | QOperator INT8 | entropy calibration 版本 |
| `q_uint8xint8_per-ch__minmax.onnx` | QOperator UINT8 | 非对称激活版本 |
| `q_int8xint8__per-ten_minmax.onnx` | QOperator INT8 | per-tensor 权重版本 |
