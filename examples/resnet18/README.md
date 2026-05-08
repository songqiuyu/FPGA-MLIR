# Example: ResNet-18 端到端编译

将 ResNet-18 INT8 量化模型从 ONNX 编译到 FPGA VLIW 指令流的完整示例。

## 前提条件

**C++ 编译器**（从仓库根目录构建）：
```bash
cd ../..
./build.sh          # 需先构建 llvm-project，见顶层 README
```

**Python 依赖**：
```bash
pip install onnx onnxruntime numpy
```

**数据文件**（放在 `examples/resnet18/data/` 下）：
```
examples/resnet18/data/resnet18.onnx                 ← 浮点模型（需手动放入）
examples/resnet18/data/resnet18_quant_int8.onnx      ← 量化模型（export_onnx.py 输出）
examples/resnet18/data/calibration/*.npy             ← 标定数据（可选）
```

---

## 运行流程

### 量化导出（若已有 INT8 ONNX 可跳过）

```bash
python export_onnx.py
# 输出：examples/resnet18/data/resnet18_quant_int8.onnx
```

### ONNX → COA MLIR

```bash
python import_mlir.py
# 输出：model/resnet18.mlir
```

### COA MLIR → VLIW 二进制

```bash
bash compile.sh
# 调用 coa-compiler，运行完整 pass 流水线：
#   shape-infer → op-fusion → tiling → addr-assign → legalize → vliw-gen
# 输出：output/resnet18.vliw
```

### 验证输出

```bash
python verify.py
# 将 output/resnet18.vliw 与 Python 参考实现逐字节对比
# 输出：output/verify_report.txt
```

---

## 目录结构

```
resnet18/
├── README.md
├── export_onnx.py       # ResNet-18 PTQ 量化 + ONNX 导出
├── import_mlir.py       # ONNX → COA MLIR 转换
├── compile.sh           # coa-compiler 编译脚本（Linux）
├── compile.bat          # coa-compiler 编译脚本（Windows）
├── verify.py            # VLIW 二进制验证
├── model/
│   ├── resnet18.mlir        # 生成的 COA MLIR（import_mlir.py 输出）
│   └── resnet18_stub.mlir   # 手写参考片段（前 3 层）
└── output/
    ├── resnet18.vliw        # 编译输出（compile.sh 输出）
    └── verify_report.txt    # 验证报告（verify.py 输出）
```

---

## 预期输出

`compile.sh` 成功时 `coa-compiler` 应打印：
```
[COAShapeInfer] Done
[COAOpFusion]   Fused 8 Conv+Add pairs
[COATiling]     All tiles legal
[COAAddrAssign] Weight DDR: 0x08000000 - 0x0BXXXXXX
[COALegalize]   All 21 ops passed
[COAVLIWGen]    Written 21 x 64 bytes -> output/resnet18.vliw
```

`verify.py` 成功时应打印：
```
[Verify] 21/21 instructions match ✓
```
