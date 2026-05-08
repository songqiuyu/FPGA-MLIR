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
optimizer/   ← AI 优化（贝叶斯 Tiling / DQN / FusionGAT / NSGA-II AutoQuant）
tests/       ← 单元测试（26/26 passing）
legacy/      ← 原 Python 参考流水线（存档）
```

## 当前状态

| 阶段 | 状态 |
|---|---|
| Phase 1: COA 方言 + 编译器流水线（C++） | ✅ 完成 |
| Phase 2: 测试体系（26 tests） | ✅ 完成 |
| Phase 3: AI 优化模块（Tiling / 融合 / 量化） | ✅ 完成 |
| M8: 构建 llvm-project + 端到端验证 | 🔲 待完成 |

## 构建

```bat
cd compiler && build.bat   # 需先构建 llvm-project，见 docs/README.md §9
```

## 测试

```bash
G:\Anaconda3\envs\502\python.exe -m pytest tests/ -v
```

---
> 详细文档见 [docs/README.md](docs/README.md)
