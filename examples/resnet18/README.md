# Example: ResNet-18 端到端编译

本示例演示将 ResNet-18 INT8 量化模型从 ONNX 编译到 FPGA VLIW 指令流的完整流程。

## 前提条件

```
FPGA-MLIR/
├── compiler/build/bin/Release/coa-compiler.exe   ← 需先构建
├── legacy/models/onnx_models/resnet18_quant_int8.onnx
└── legacy/datasets/calibration_data/             ← 标定数据
```

Python 环境（conda 502）：
```bash
pip install onnx onnxruntime numpy
```

---

## 编译流程概览

```
resnet18_quant_int8.onnx
        │
        ▼  Step 1: 01_export_onnx.py
resnet18_quant_int8.onnx  （若尚未量化，则执行量化导出）
        │
        ▼  Step 2: 02_import_mlir.py
model/resnet18.mlir        （Level-1 COA MLIR：ONNX 量化参数 + 算子图）
        │
        ▼  Step 3: 03_compile.bat  (coa-compiler C++ pipeline)
        │   --coa-shape-infer
        │   --coa-op-fusion
        │   --coa-tiling
        │   --coa-addr-assign
        │   --coa-legalize
        │   --coa-vliw-gen
        ▼
output/resnet18.vliw       （512-bit VLIW 二进制，每层一条指令）
        │
        ▼  Step 4: 04_verify.py
output/verify_report.txt   （与 Python 参考输出逐字节对比）
```

---

## 快速运行

```bat
REM Step 1: 量化并导出 ONNX（若已有量化模型可跳过）
G:\Anaconda3\envs\502\python.exe 01_export_onnx.py

REM Step 2: 导入为 COA MLIR
G:\Anaconda3\envs\502\python.exe 02_import_mlir.py

REM Step 3: 编译到 VLIW（需已构建 coa-compiler）
03_compile.bat

REM Step 4: 验证输出
G:\Anaconda3\envs\502\python.exe 04_verify.py
```

---

## 目录结构

```
resnet18/
├── README.md
├── 01_export_onnx.py    # ResNet-18 PTQ 量化 + ONNX 导出
├── 02_import_mlir.py    # ONNX → COA MLIR 转换
├── 03_compile.bat       # coa-compiler 编译脚本
├── 04_verify.py         # VLIW 二进制验证
├── model/
│   ├── resnet18.mlir    # 生成的 COA MLIR（Step 2 输出）
│   └── resnet18_stub.mlir   # 手写参考片段（前 3 层）
└── output/
    ├── resnet18.vliw    # 编译输出（Step 3 输出）
    └── verify_report.txt
```

---

## 预期输出

Step 3 成功时 `coa-compiler` 应打印：
```
[COACompiler] Parsed 21 ops from model/resnet18.mlir
[COAShapeInfer] Done
[COAOpFusion]   Fused 8 Conv+Add pairs
[COATiling]     All tiles legal
[COAAddrAssign] Weight DDR: 0x08000000 - 0x0BXXXXXX
[COALegalize]   All 21 ops passed
[COAVLIWGen]    Written 21 x 64 bytes -> output/resnet18.vliw
```

Step 4 成功时应打印：
```
[Verify] Comparing output/resnet18.vliw vs Python reference...
[Verify] 21/21 instructions match ✓
```
