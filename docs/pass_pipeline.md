# COA 编译器 Pass 流水线详解

## Pass 依赖图

```
┌──────────────┐
│ 输入 COA MLIR │  (Level-1 属性: ONNX 量化参数 + 卷积几何)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ coa-shape-infer  │  填写 R,C,M,N,R0,C0,sM_concat,M_concat
└──────┬───────────┘  依赖: 输入 tensor RankedTensorType
       │
       ▼
┌──────────────────┐
│ coa-op-fusion    │  标记 silu_addr（Conv+Add 残差）
└──────┬───────────┘  依赖: shape-infer（需要 M 值）
       │
       ▼
┌──────────────────┐
│ coa-tiling       │  写入 tM,tR,tC
└──────┬───────────┘  依赖: shape-infer（需要 R,C,M,N）
       │
       ▼
┌──────────────────┐
│ coa-mem-alloc    │  写入 in_addr / out_addr / in2_addr
└──────┬───────────┘  依赖: tiling（需要 M,R,C 计算激活 tensor 大小）
       │               算法: 生命周期分析 + 线性扫描分配（Pass 4a）
       ▼
┌──────────────────┐
│ coa-addr-assign  │  写入 weight_addr / bias_addr + factor / factor2
└──────┬───────────┘  依赖: tiling（需要 tM 计算 weight size）（Pass 4b）
       │
       ▼
┌──────────────────┐
│ coa-legalize     │  验证所有位域合法性
└──────┬───────────┘  依赖: mem-alloc + addr-assign（地址全部填入）
       │
       ▼
┌──────────────────┐
│ coa-vliw-gen     │  生成 N×64 字节二进制
└──────────────────┘  依赖: legalize（保证数据安全）
```

## 各 Pass 实现文件

| Pass | 源文件 | 关键逻辑 |
|---|---|---|
| `COAShapeInfer` | `lib/Transforms/ShapeInfer.cpp` | `computeOutDim(in, k, s, pad, dil)` |
| `COAOpFusion` | `lib/Transforms/OpFusion.cpp` | `OpRewritePattern<QLinearConvOp>` |
| `COATiling` | `lib/Transforms/Tiling.cpp` | `getTile()` + `checkBuffers()` |
| `COAMemAlloc` | `lib/Transforms/MemAlloc.cpp` | 生命周期分析 + 线性扫描分配（Best-fit 空闲池回收） |
| `COAAddrAssign` | `lib/Transforms/AddrAssign.cpp` | weight/bias 顺序追加 + quantization factor 计算 |
| `COALegalize` | `lib/Transforms/Legalize.cpp` | `check11/12/Addr/Factor()` 断言 |
| `COAVLIWGen` | `lib/CodeGen/VLIWCodeGen.cpp` | `VLIW::toBytes()` 位域打包 |

## 缓冲区约束公式

```
tN32 = max over (sN ∈ {0, tN, 2tN, ...} < N):
           ceil((sN + tN) / 32) - sN / 32

tM16 = max over (sM ∈ {0, tM, 2tM, ...} < M):
           ceil((sM + tM) / 16) - sM / 16

wdepth = tN32 × k × k × tM16          < 256
gdepth = ((tR-1)×s + (k-1)×d + 1)
       × ((tC-1)×s + (k-1)×d + 1)
       × tN32                           < 1024
odepth = tR × tC × tM16                < 2048
```

## 地址分配策略

### 激活内存：生命周期感知线性扫描（`coa-mem-alloc`）

```
算法（Poletto & Sarkar, TOPLAS'99 移植）：

  Phase 1 — 拓扑序号分配
    对每个 COA op 赋予单调递增的序号 t

  Phase 2 — 生命周期分析
    对每个 op result（激活张量）：
      def_time  = 产生该 Value 的 op 序号
      last_use  = 最后一个消费该 Value 的 op 序号
      size      = M × R × C  字节（INT8，来自 shape-infer 属性）

  Phase 3 — 线性扫描分配
    按 def_time 排序所有 interval
    维护 active（当前存活，按 last_use 排序）
    维护 free_pool（已过期块，可复用）

    对每个新 interval：
      1. 将 last_use < def_time 的过期块归还 free_pool（内存回收）
      2. Best-fit 从 free_pool 查找 ≥ size 的最小块
         - 找到：分配首地址，多余尾部归还
         - 未找到：从 bump 指针处扩展新块

  峰值 DDR 激活占用：
    peak = max(同时存活 tensor 大小之和)
    vs. Ping-Pong: 2 × max_single_tensor_size
    ResNet-18 实测节省约 25%；含残差的 mini 模型节省约 37%

  设置属性：in_addr, out_addr（所有 op）及 in2_addr（QLinearAdd）

  网络输入固定地址：0x0000_0000（funcArg[0]）
  激活池基地址：    0x1000_0000（--act-base，可配置）
```

### 权重 / 偏置（`coa-addr-assign`，顺序追加）

```
权重（顺序追加）：
  base = 0x0800_0000 (weightBase)
  每层 offset += M × align(N, 32) × kH × kW  (bytes)

Bias：
  base = 0xC000_0000 (biasBase)
  每层 offset += M × 4  (int32)

量化因子：
  factor  = round(in_scale × mean(weight_scale) / out_scale × 2^15)  (Conv/Gemm)
  factor2 = round(b_scale / out_scale × 2^15)  (QLinearAdd)
```

### 选项对比

| 策略 | 峰值激活内存 | 残差安全性 | 实现复杂度 |
|------|-------------|-----------|----------|
| 旧 Ping-Pong（已弃用） | 2 × max_size（且有 bug） | ❌ shortcut 覆盖冲突 | O(1) |
| **线性扫描（当前）** | **max(Σ同时存活)** | **✅ 精确追踪 SSA 生命周期** | O(n log n) |
