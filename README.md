# FPGA-MLIR AI 编译器

> 面向自定义 FPGA 粗粒度 VLIW 指令集的 AI 编译器。  
> 输入 ONNX 量化模型 → 输出 512-bit VLIW 二进制指令流。

## 快速导航

| 文档 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 完整项目文档（架构 / 方言 / 指令结构 / 用法） |
| [docs/pass_pipeline.md](docs/pass_pipeline.md) | 编译器 Pass 流水线详解 |
| [docs/ai_optimizer.md](docs/ai_optimizer.md) | AI 优化模块（Tiling / GNN融合 / AutoQuant）详解 |

## 核心模块

```
compiler/    ← C++ MLIR 编译器（TableGen ODS + 6 Passes + coa-opt/coa-compiler）
coa/         ← Python 前端库（onnx_importer / mlir_parser / vliw / tiling）
optimizer/   ← AI 优化（贝叶斯 Tiling / DQN / FusionGAT / NSGA-II AutoQuant）
examples/    ← 端到端示例（ResNet-18 ONNX → VLIW，已验证 30 条指令）
tests/       ← 单元测试（26/26 passing）
```

## 当前状态

| 阶段 | 状态 |
|---|---|
| Phase 1: COA 方言 + 编译器流水线（C++） | ✅ 完成 |
| Phase 2: 测试体系（26 tests） | ✅ 完成 |
| Phase 3: AI 优化模块（Tiling / 融合 / 量化） | ✅ 完成 |
| M8: LLVM-15 + onnx_importer 修复 + ResNet-18 端到端验证 | ✅ 完成 |
| M9: verify.py Python 参考对齐 | 🔲 计划中 |

## 构建

```bash
# 使用系统已安装的 LLVM-15（Linux）
cmake -S compiler -B compiler/build \
      -DMLIR_DIR=/usr/lib/llvm-15/lib/cmake/mlir \
      -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm
cmake --build compiler/build -j$(nproc)
# 产物：compiler/build/bin/coa-compiler, compiler/build/bin/coa-opt
```

## 测试

```bash
# Python 单元测试（直接运行）
python -m pytest tests/ -v

# 通过 CTest 运行（构建后）
ctest --test-dir build -V --label-regex python
```

---
> 详细文档见 [docs/README.md](docs/README.md)
