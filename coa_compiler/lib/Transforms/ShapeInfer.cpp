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
#include "COA/COAPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::coa {

#define GEN_PASS_DEF_COASHAPEINFER
#include "COA/COAPasses.h.inc"

namespace {

/// Compute output spatial dimension: (in + pad_begin + pad_end - (k-1)*d - 1) / s + 1
static int64_t computeOutDim(int64_t in, int64_t k, int64_t s, int64_t pad_begin,
                              int64_t pad_end, int64_t d) {
    return (in + pad_begin + pad_end - (k - 1) * d - 1) / s + 1;
}

/// Try to get ranked tensor shape; return empty if unranked.
static SmallVector<int64_t> getShape(Value v) {
    if (auto rt = dyn_cast<RankedTensorType>(v.getType()))
        return SmallVector<int64_t>(rt.getShape().begin(), rt.getShape().end());
    return {};
}

struct COAShapeInferPass : public impl::COAShapeInferBase<COAShapeInferPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        OpBuilder builder(funcOp.getContext());

        funcOp.walk([&](Operation *op) {
            builder.setInsertionPoint(op);

            if (auto conv = dyn_cast<QLinearConvOp>(op)) {
                auto inShape = getShape(conv.getInput());
                if (inShape.size() != 4) return;

                // inShape: [N_batch, C_in, H_in, W_in]
                int64_t H_in = inShape[2], W_in = inShape[3];
                int64_t C_in = inShape[1];

                auto kernel   = conv.getKernelShape();
                auto strides  = conv.getStrides();
                auto pads     = conv.getPads();
                auto dilations = conv.getDilations();

                int64_t kH = kernel[0], kW = kernel[1];
                int64_t sH = strides[0], sW = strides[1];
                int64_t pH0 = pads[0], pW0 = pads[1], pH1 = pads[2], pW1 = pads[3];
                int64_t dH = dilations[0], dW = dilations[1];

                int64_t H_out = computeOutDim(H_in, kH, sH, pH0, pH1, dH);
                int64_t W_out = computeOutDim(W_in, kW, sW, pW0, pW1, dW);

                // Output channels M from weight shape [M, C_in/group, kH, kW]
                int64_t C_out = 0;
                auto wShape = getShape(conv.getWeight());
                if (!wShape.empty()) C_out = wShape[0];

                conv->setAttr("R",  builder.getI64IntegerAttr(H_out));
                conv->setAttr("C",  builder.getI64IntegerAttr(W_out));
                conv->setAttr("M",  builder.getI64IntegerAttr(C_out));
                conv->setAttr("N",  builder.getI64IntegerAttr(C_in));
                conv->setAttr("R0", builder.getI64IntegerAttr(H_in));
                conv->setAttr("C0", builder.getI64IntegerAttr(W_in));

                // sM_concat and M_concat default to M (no concat)
                if (conv.getSMConcat() == 0)
                    conv->setAttr("sM_concat", builder.getI64IntegerAttr(C_out));
                if (conv.getMConcat() == 0)
                    conv->setAttr("M_concat",  builder.getI64IntegerAttr(C_out));

            } else if (auto pool = dyn_cast<MaxPoolOp>(op)) {
                auto inShape = getShape(pool.getInput());
                if (inShape.size() != 4) return;

                int64_t H_in = inShape[2], W_in = inShape[3], C_in = inShape[1];
                auto kernel  = pool.getKernelShape();
                auto strides = pool.getStrides();
                auto pads    = pool.getPads();

                int64_t kH = kernel[0], kW = kernel[1];
                int64_t sH = strides[0], sW = strides[1];
                int64_t pH = pads[0], pW = pads[1];

                int64_t H_out = computeOutDim(H_in, kH, sH, pH, pads[2], 1);
                int64_t W_out = computeOutDim(W_in, kW, sW, pW, pads[3], 1);

                pool->setAttr("R",  builder.getI64IntegerAttr(H_out));
                pool->setAttr("C",  builder.getI64IntegerAttr(W_out));
                pool->setAttr("M",  builder.getI64IntegerAttr(C_in));
                pool->setAttr("N",  builder.getI64IntegerAttr(C_in));
                pool->setAttr("R0", builder.getI64IntegerAttr(H_in));
                pool->setAttr("C0", builder.getI64IntegerAttr(W_in));

            } else if (auto add = dyn_cast<QLinearAddOp>(op)) {
                auto inShape = getShape(add.getA());
                if (inShape.size() != 4) return;
                add->setAttr("R",  builder.getI64IntegerAttr(inShape[2]));
                add->setAttr("C",  builder.getI64IntegerAttr(inShape[3]));
                add->setAttr("M",  builder.getI64IntegerAttr(inShape[1]));
                add->setAttr("N",  builder.getI64IntegerAttr(inShape[1]));
                add->setAttr("R0", builder.getI64IntegerAttr(inShape[2]));
                add->setAttr("C0", builder.getI64IntegerAttr(inShape[3]));

            } else if (auto gap = dyn_cast<QLinearGlobalAveragePoolOp>(op)) {
                auto inShape = getShape(gap.getInput());
                if (inShape.size() != 4) return;
                gap->setAttr("R",  builder.getI64IntegerAttr(1));
                gap->setAttr("C",  builder.getI64IntegerAttr(1));
                gap->setAttr("M",  builder.getI64IntegerAttr(inShape[1]));
                gap->setAttr("N",  builder.getI64IntegerAttr(inShape[1]));
                gap->setAttr("R0", builder.getI64IntegerAttr(inShape[2]));
                gap->setAttr("C0", builder.getI64IntegerAttr(inShape[3]));

            } else if (auto gemm = dyn_cast<QGemmOp>(op)) {
                // Treat GEMM as 1x1 conv: [batch, N] x [M, N]^T -> [batch, M]
                auto aShape = getShape(gemm.getA());
                auto bShape = getShape(gemm.getB());
                if (aShape.empty() || bShape.empty()) return;

                int64_t N_in = aShape.back();
                int64_t M_out = bShape[0]; // assuming transB=1

                gemm->setAttr("R",  builder.getI64IntegerAttr(1));
                gemm->setAttr("C",  builder.getI64IntegerAttr(1));
                gemm->setAttr("M",  builder.getI64IntegerAttr(M_out));
                gemm->setAttr("N",  builder.getI64IntegerAttr(N_in));
                gemm->setAttr("R0", builder.getI64IntegerAttr(1));
                gemm->setAttr("C0", builder.getI64IntegerAttr(1));

                if (gemm.getSMConcat() == 0)
                    gemm->setAttr("sM_concat", builder.getI64IntegerAttr(M_out));
                if (gemm.getMConcat() == 0)
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
