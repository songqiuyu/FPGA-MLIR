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
│ coa-addr-assign  │  写入 in/out/weight/bias_addr + factor
└──────┬───────────┘  依赖: tiling（需要 tM 计算 weight size）
       │
       ▼
┌──────────────────┐
│ coa-legalize     │  验证所有位域合法性
└──────┬───────────┘  依赖: addr-assign（地址全部填入）
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
| `COAAddrAssign` | `lib/Transforms/AddrAssign.cpp` | Ping-pong 激活 + weight offset 累积 |
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

```
激活缓冲（Ping-Pong）：
  第一层输入: 0x0000_0000
  后续激活:   0x1000_0000  (activationBase, 可配置)

权重（顺序追加）：
  base = 0x0800_0000 (weightBase)
  每层 offset += M × align(N, 32) × kH × kW  (bytes)

Bias：
  base = 0xC000_0000 (biasBase)
  每层 offset += M × 4  (int32)
```
