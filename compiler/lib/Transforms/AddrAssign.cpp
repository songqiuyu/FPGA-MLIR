//===- AddrAssign.cpp - COA weight/bias address + factor pass ------*- C++ -*-===//
//
// Pass: --coa-addr-assign
//
// Assigns weight_addr / bias_addr to each COA op using sequential append,
// and computes the fixed-point quantization factor (factor / factor2).
//
// NOTE: Activation addresses (in_addr, out_addr, in2_addr) are now assigned
// by --coa-mem-alloc, which must run before this pass.
//
// Weight layout:  weightBase + Σ(M × N_pad32 × kH × kW)  (INT8)
// Bias layout:    biasBase   + Σ(M × 4)                   (INT32)
// Factor:         round(in_scale × weight_scale_mean / out_scale × 2^15)
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <cmath>
#include <cstdint>

namespace mlir::coa {

#define GEN_PASS_CLASSES
#include "COA/COAPasses.h.inc"

namespace {

static double toDouble(llvm::APFloat f) { return f.convertToDouble(); }

static SmallVector<double> arrF(ArrayAttr a) {
    SmallVector<double> v;
    for (auto e : a) v.push_back(e.cast<FloatAttr>().getValueAsDouble());
    return v;
}

static int64_t arrI(ArrayAttr a, unsigned i, int64_t def = 0) {
    if (!a || i >= a.size()) return def;
    return a[i].cast<IntegerAttr>().getInt();
}

static int64_t alignUp(int64_t x, int64_t align) {
    return ((x + align - 1) / align) * align;
}

/// factor = round(inScale × mean(wScales) / outScale × 2^15)
static int64_t computeConvFactor(double inScale,
                                 const SmallVector<double> &wScales,
                                 double outScale) {
    double wMean = 0.0;
    for (double s : wScales) wMean += s;
    if (!wScales.empty()) wMean /= static_cast<double>(wScales.size());
    return static_cast<int64_t>(std::round(inScale * wMean / outScale * (1LL << 15)));
}

/// factor  = round(aScale / outScale × 2^15)
/// factor2 = round(bScale / outScale × 2^15)
static std::pair<int64_t, int64_t>
computeAddFactors(double aScale, double bScale, double outScale) {
    return {static_cast<int64_t>(std::round(aScale / outScale * (1LL << 15))),
            static_cast<int64_t>(std::round(bScale / outScale * (1LL << 15)))};
}

struct COAAddrAssignPass : public COAAddrAssignBase<COAAddrAssignPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        OpBuilder builder(funcOp.getContext());

        int64_t wBase = weightBase;
        int64_t bBase = biasBase;

        int64_t weightOffset = 0;
        int64_t biasOffset   = 0;

        funcOp.walk([&](Operation *op) {
            builder.setInsertionPoint(op);

            if (auto conv = dyn_cast<QLinearConvOp>(op)) {
                int64_t N  = alignUp(conv.N(), 32);
                int64_t M  = conv.M();
                auto kernel = conv.kernel_shape();
                int64_t kH = arrI(kernel, 0, 1), kW = arrI(kernel, 1, 1);

                int64_t wSize = M * N * kH * kW; // INT8 weights
                int64_t bSize = M * 4;            // INT32 bias

                conv->setAttr("weight_addr",
                    builder.getI64IntegerAttr(wBase + weightOffset));
                conv->setAttr("bias_addr",
                    builder.getI64IntegerAttr(bBase + biasOffset));

                SmallVector<double> wScales = arrF(conv.weight_scale());
                conv->setAttr("factor", builder.getI64IntegerAttr(
                    computeConvFactor(toDouble(conv.in_scale()), wScales,
                                      toDouble(conv.out_scale()))));

                weightOffset += wSize;
                biasOffset   += bSize;

            } else if (auto gemm = dyn_cast<QGemmOp>(op)) {
                int64_t N = alignUp(gemm.N(), 32);
                int64_t M = alignUp(gemm.M(), 16);

                int64_t wSize = M * N;
                int64_t bSize = gemm.M() * 4;

                gemm->setAttr("weight_addr",
                    builder.getI64IntegerAttr(wBase + weightOffset));
                gemm->setAttr("bias_addr",
                    builder.getI64IntegerAttr(bBase + biasOffset));

                SmallVector<double> bScales = arrF(gemm.b_scale());
                gemm->setAttr("factor", builder.getI64IntegerAttr(
                    computeConvFactor(toDouble(gemm.a_scale()), bScales,
                                      toDouble(gemm.out_scale()))));

                weightOffset += wSize;
                biasOffset   += bSize;

            } else if (auto add = dyn_cast<QLinearAddOp>(op)) {
                auto [f, f2] = computeAddFactors(
                    toDouble(add.a_scale()), toDouble(add.b_scale()),
                    toDouble(add.out_scale()));
                add->setAttr("factor",  builder.getI64IntegerAttr(f));
                add->setAttr("factor2", builder.getI64IntegerAttr(f2));

                // in_addr / in2_addr / out_addr already set by coa-mem-alloc.
            }
            // MaxPool and QLinearGlobalAveragePool have no weight/bias/factor.
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOAAddrAssignPass() {
    return std::make_unique<COAAddrAssignPass>();
}

} // namespace mlir::coa
