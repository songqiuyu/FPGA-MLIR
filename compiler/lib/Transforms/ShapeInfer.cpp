//===- ShapeInfer.cpp - COA shape inference pass -------------------*- C++ -*-===//
//
// Pass: --coa-shape-infer
//
// Walks the function and fills in R, C, M, N, R0, C0 dimension attributes on
// each COA op by computing output shapes from input tensor types and convolution
// geometry (kernel, stride, pad, dilation).
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::coa {

#define GEN_PASS_CLASSES
#include "COA/COAPasses.h.inc"

namespace {

/// Extract i-th int64 from an ArrayAttr (I64ArrayAttr element).
static int64_t arrI(ArrayAttr a, unsigned i, int64_t def = 0) {
    if (!a || i >= a.size()) return def;
    return a[i].cast<IntegerAttr>().getInt();
}
/// Extract i-th int64 from an optional ArrayAttr.
static int64_t optI(llvm::Optional<::mlir::ArrayAttr> a, unsigned i, int64_t def = 0) {
    if (!a) return def;
    return arrI(*a, i, def);
}

/// Compute output spatial dimension: (in + pad_begin + pad_end - (k-1)*d - 1) / s + 1
static int64_t computeOutDim(int64_t in, int64_t k, int64_t s, int64_t pad_begin,
                              int64_t pad_end, int64_t d) {
    return (in + pad_begin + pad_end - (k - 1) * d - 1) / s + 1;
}

/// Try to get ranked tensor shape; return empty if unranked.
static SmallVector<int64_t> getShape(Value v) {
    if (auto rt = v.getType().dyn_cast<RankedTensorType>())
        return SmallVector<int64_t>(rt.getShape().begin(), rt.getShape().end());
    return {};
}

struct COAShapeInferPass : public COAShapeInferBase<COAShapeInferPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        OpBuilder builder(funcOp.getContext());

        funcOp.walk([&](Operation *op) {
            builder.setInsertionPoint(op);

            if (auto conv = dyn_cast<QLinearConvOp>(op)) {
                auto inShape = getShape(conv.input());
                if (inShape.size() != 4) return;

                // inShape: [N_batch, C_in, H_in, W_in]
                int64_t H_in = inShape[2], W_in = inShape[3];
                int64_t C_in = inShape[1];

                auto kernel   = conv.kernel_shape();
                auto strides  = conv.strides();
                auto pads     = conv.pads();
                auto dilations = conv.dilations();

                int64_t kH = arrI(kernel,0,1),   kW = arrI(kernel,1,1);
                int64_t sH = optI(strides,0,1),   sW = optI(strides,1,1);
                int64_t pH0 = optI(pads,0,0), pW0 = optI(pads,1,0),
                        pH1 = optI(pads,2,0), pW1 = optI(pads,3,0);
                int64_t dH = optI(dilations,0,1), dW = optI(dilations,1,1);

                int64_t H_out = computeOutDim(H_in, kH, sH, pH0, pH1, dH);
                int64_t W_out = computeOutDim(W_in, kW, sW, pW0, pW1, dW);

                // Output channels M from weight shape [M, C_in/group, kH, kW]
                int64_t C_out = 0;
                auto wShape = getShape(conv.weight());
                if (!wShape.empty()) C_out = wShape[0];

                conv->setAttr("R",  builder.getI64IntegerAttr(H_out));
                conv->setAttr("C",  builder.getI64IntegerAttr(W_out));
                conv->setAttr("M",  builder.getI64IntegerAttr(C_out));
                conv->setAttr("N",  builder.getI64IntegerAttr(C_in));
                conv->setAttr("R0", builder.getI64IntegerAttr(H_in));
                conv->setAttr("C0", builder.getI64IntegerAttr(W_in));

                // sM_concat and M_concat default to M (no concat)
                if (conv.sM_concat() == 0)
                    conv->setAttr("sM_concat", builder.getI64IntegerAttr(C_out));
                if (conv.M_concat() == 0)
                    conv->setAttr("M_concat",  builder.getI64IntegerAttr(C_out));

            } else if (auto pool = dyn_cast<MaxPoolOp>(op)) {
                auto inShape = getShape(pool.input());
                if (inShape.size() != 4) return;

                int64_t H_in = inShape[2], W_in = inShape[3], C_in = inShape[1];
                auto kernel  = pool.kernel_shape();
                auto strides = pool.strides();
                auto pads    = pool.pads();

                int64_t kH = arrI(kernel,0,2), kW = arrI(kernel,1,2);
                int64_t sH = optI(strides,0,2), sW = optI(strides,1,2);
                int64_t pH0 = optI(pads,0,0), pW0 = optI(pads,1,0),
                        pH1 = optI(pads,2,0), pW1 = optI(pads,3,0);

                int64_t H_out = computeOutDim(H_in, kH, sH, pH0, pH1, 1);
                int64_t W_out = computeOutDim(W_in, kW, sW, pW0, pW1, 1);

                pool->setAttr("R",  builder.getI64IntegerAttr(H_out));
                pool->setAttr("C",  builder.getI64IntegerAttr(W_out));
                pool->setAttr("M",  builder.getI64IntegerAttr(C_in));
                pool->setAttr("N",  builder.getI64IntegerAttr(C_in));
                pool->setAttr("R0", builder.getI64IntegerAttr(H_in));
                pool->setAttr("C0", builder.getI64IntegerAttr(W_in));

            } else if (auto add = dyn_cast<QLinearAddOp>(op)) {
                auto inShape = getShape(add.a());
                if (inShape.size() != 4) return;
                add->setAttr("R",  builder.getI64IntegerAttr(inShape[2]));
                add->setAttr("C",  builder.getI64IntegerAttr(inShape[3]));
                add->setAttr("M",  builder.getI64IntegerAttr(inShape[1]));
                add->setAttr("N",  builder.getI64IntegerAttr(inShape[1]));
                add->setAttr("R0", builder.getI64IntegerAttr(inShape[2]));
                add->setAttr("C0", builder.getI64IntegerAttr(inShape[3]));

            } else if (auto gap = dyn_cast<QLinearGlobalAveragePoolOp>(op)) {
                auto inShape = getShape(gap.input());
                if (inShape.size() != 4) return;
                gap->setAttr("R",  builder.getI64IntegerAttr(1));
                gap->setAttr("C",  builder.getI64IntegerAttr(1));
                gap->setAttr("M",  builder.getI64IntegerAttr(inShape[1]));
                gap->setAttr("N",  builder.getI64IntegerAttr(inShape[1]));
                gap->setAttr("R0", builder.getI64IntegerAttr(inShape[2]));
                gap->setAttr("C0", builder.getI64IntegerAttr(inShape[3]));

            } else if (auto gemm = dyn_cast<QGemmOp>(op)) {
                // Treat GEMM as 1x1 conv: [batch, N] x [M, N]^T -> [batch, M]
                auto aShape = getShape(gemm.a());
                auto bShape = getShape(gemm.b());
                if (aShape.empty() || bShape.empty()) return;

                int64_t N_in = aShape.back();
                int64_t M_out = bShape[0]; // assuming transB=1

                gemm->setAttr("R",  builder.getI64IntegerAttr(1));
                gemm->setAttr("C",  builder.getI64IntegerAttr(1));
                gemm->setAttr("M",  builder.getI64IntegerAttr(M_out));
                gemm->setAttr("N",  builder.getI64IntegerAttr(N_in));
                gemm->setAttr("R0", builder.getI64IntegerAttr(1));
                gemm->setAttr("C0", builder.getI64IntegerAttr(1));

                if (gemm.sM_concat() == 0)
                    gemm->setAttr("sM_concat", builder.getI64IntegerAttr(M_out));
                if (gemm.M_concat() == 0)
                    gemm->setAttr("M_concat",  builder.getI64IntegerAttr(M_out));
            }
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOAShapeInferPass() {
    return std::make_unique<COAShapeInferPass>();
}

} // namespace mlir::coa
