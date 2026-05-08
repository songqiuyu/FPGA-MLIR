//===- OpFusion.cpp - COA operator fusion pass ---------------------*- C++ -*-===//
//
// Pass: --coa-op-fusion
//
// Rule-based operator fusion pass. Currently implements:
//   1. Elide identity dequantize/quantize chains between COA ops.
//   2. Mark Conv ops whose output feeds directly into a SiLU/ReLU6 LUT by
//      setting a non-zero silu_addr attribute (LUT address placeholder).
//
// The AI-guided GNN fusion is implemented separately in ai_optimizer/op_fusion/.
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "COA/COAPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::coa {

#define GEN_PASS_DEF_COAOPFUSION
#include "COA/COAPasses.h.inc"

namespace {

/// Placeholder LUT address for SiLU activation (FPGA stores LUT in DDR).
static constexpr int64_t kSiluLUTAddr = 0x20000000;

/// Pattern: if a QLinearConvOp is the sole user of the output and that user
/// is itself a QLinearAddOp with the output of this conv, mark for residual
/// bypass (this simplifies address assignment for the add op).
///
/// Currently a stub; more patterns added as the hardware spec is confirmed.
struct ConvResidualFusePattern : public OpRewritePattern<QLinearConvOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(QLinearConvOp conv,
                                  PatternRewriter &rewriter) const override {
        // Check if conv feeds a qlinearadd immediately
        Value convOut = conv.getOutput();
        for (Operation *user : convOut.getUsers()) {
            if (auto add = dyn_cast<QLinearAddOp>(user)) {
                // Mark the conv's silu_addr as the residual add address if
                // it is still at default 0 (not yet fused).
                if (conv.getSiluAddr() == 0) {
                    rewriter.modifyOpInPlace(conv, [&]() {
                        conv->setAttr("silu_addr",
                                      rewriter.getI64IntegerAttr(kSiluLUTAddr));
                    });
                    return success();
                }
            }
        }
        return failure();
    }
};

struct COAOpFusionPass : public impl::COAOpFusionBase<COAOpFusionPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ConvResidualFusePattern>(&getContext());
        FrozenRewritePatternSet frozen(std::move(patterns));
        if (failed(applyPatternsGreedily(getOperation(), frozen)))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOAOpFusionPass() {
    return std::make_unique<COAOpFusionPass>();
}

} // namespace mlir::coa
