# Examples

每个子目录是一个独立的端到端编译示例，按难度递进。

| 示例 | 模型 | 状态 | 说明 |
|---|---|---|---|
| [resnet18/](resnet18/) | ResNet-18 INT8 | 🔲 框架完成，待 coa-compiler 构建后验证 | 图像分类，21 层，标准 VLIW 编译流程 |
| yolo10/ | YOLOv10-n INT8 | 🔲 计划中 | 目标检测，含 CSP / C2f 结构 |
| bert_tiny/ | BERT-tiny INT8 | 🔲 计划中 | Transformer，含 GEMM 密集层 |

---

## 通用依赖

```bash
# Python 环境（conda 502）
pip install onnx onnxruntime numpy

# C++ 编译器（需先构建）
cd compiler && build.bat
```

## 运行任意示例

```bat
cd examples\resnet18
G:\Anaconda3\envs\502\python.exe 01_export_onnx.py
G:\Anaconda3\envs\502\python.exe 02_import_mlir.py
03_compile.bat
G:\Anaconda3\envs\502\python.exe 04_verify.py
```
