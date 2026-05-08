//===- AddrAssign.cpp - COA DDR address assignment pass ------------*- C++ -*-===//
//
// Pass: --coa-addr-assign
//
// Assigns DDR addresses to each COA op's tensors and computes the fixed-point
// quantization factor (factor / factor2).  Ports assign_addr.py logic to C++.
//
// Address layout:
//   First layer input:  0x0
//   Activation buffers: activationBase (default 0x10000000), ping-pong
//   Weight tensors:     weightBase + cumulative offset (IC padded to 32)
//   Bias tensors:       biasBase + cumulative offset
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <cmath>
#include <cstdint>
#include <map>
#include <string>

namespace mlir::coa {

#define GEN_PASS_CLASSES
#include "COA/COAPasses.h.inc"

namespace {

static int64_t arrI(ArrayAttr a, unsigned i, int64_t def = 0) {
    if (!a || i >= a.size()) return def;
    return a[i].cast<IntegerAttr>().getInt();
}
static double toDouble(llvm::APFloat f) {
    return f.convertToDouble();
}
static SmallVector<double> arrF(ArrayAttr a) {
    SmallVector<double> v;
    for (auto e : a) v.push_back(e.cast<FloatAttr>().getValueAsDouble());
    return v;
}

/// Round up x to the next multiple of align.
static int64_t alignUp(int64_t x, int64_t align) {
    return ((x + align - 1) / align) * align;
}

/// Compute fixed-point factor for convolution:
///   factor = round(in_scale * weight_scale[0] / out_scale * 2^15)
/// Weight scale is taken as the geometric mean for per-channel (approximate).
static int64_t computeConvFactor(double inScale, const SmallVector<double> &wScales,
                                 double outScale) {
    double wScaleMean = 0.0;
    for (double s : wScales) wScaleMean += s;
    if (!wScales.empty()) wScaleMean /= static_cast<double>(wScales.size());
    double M = (inScale * wScaleMean) / outScale;
    return static_cast<int64_t>(std::round(M * (1LL << 15)));
}

/// Compute fixed-point factor for qlinearadd:
///   factor  = round(a_scale / out_scale * 2^15)
///   factor2 = round(b_scale / out_scale * 2^15)
static std::pair<int64_t, int64_t>
computeAddFactors(double aScale, double bScale, double outScale) {
    return {static_cast<int64_t>(std::round(aScale / outScale * (1LL << 15))),
            static_cast<int64_t>(std::round(bScale / outScale * (1LL << 15)))};
}

struct COAAddrAssignPass : public COAAddrAssignBase<COAAddrAssignPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        OpBuilder builder(funcOp.getContext());

        int64_t wBase    = weightBase;
        int64_t bBase    = biasBase;
        int64_t actBase  = activationBase;

        int64_t weightOffset = 0; // cumulative weight offset in bytes
        int64_t biasOffset   = 0; // cumulative bias offset in bytes

        bool isFirstOp = true;

        funcOp.walk([&](Operation *op) {
            builder.setInsertionPoint(op);

            // Helper: get activation addresses (ping-pong)
            auto getActAddrs = [&](bool firstLayer) -> std::pair<int64_t, int64_t> {
                if (firstLayer) {
                    return {0LL, actBase};
                }
                return {actBase, actBase};
            };

            if (auto conv = dyn_cast<QLinearConvOp>(op)) {
                int64_t N = alignUp(conv.N(), 32);
                int64_t M = conv.M();
                auto kernel = conv.kernel_shape();
                int64_t kH = arrI(kernel,0,1), kW = arrI(kernel,1,1);

                // Weight size: M * N_aligned * kH * kW (int8)
                int64_t wSize = M * N * kH * kW;
                // Bias size: M * sizeof(int32) = M * 4
                int64_t bSize = M * 4;

                auto [inAddr, outAddr] = getActAddrs(isFirstOp);

                conv->setAttr("in_addr",     builder.getI64IntegerAttr(inAddr));
                conv->setAttr("out_addr",    builder.getI64IntegerAttr(outAddr));
                conv->setAttr("weight_addr", builder.getI64IntegerAttr(wBase + weightOffset));
                conv->setAttr("bias_addr",   builder.getI64IntegerAttr(bBase + biasOffset));

                // Compute quantization factor
                SmallVector<double> wScales = arrF(conv.weight_scale());
                int64_t factor = computeConvFactor(
                    toDouble(conv.in_scale()), wScales,
                    toDouble(conv.out_scale()));
                conv->setAttr("factor", builder.getI64IntegerAttr(factor));

                weightOffset += wSize;
                biasOffset   += bSize;
                isFirstOp = false;

            } else if (auto gemm = dyn_cast<QGemmOp>(op)) {
                int64_t N = alignUp(gemm.N(), 32);
                int64_t M = alignUp(gemm.M(), 16);

                int64_t wSize = M * N;
                int64_t bSize = gemm.M() * 4;

                auto [inAddr, outAddr] = getActAddrs(isFirstOp);

                gemm->setAttr("in_addr",     builder.getI64IntegerAttr(inAddr));
                gemm->setAttr("out_addr",    builder.getI64IntegerAttr(outAddr));
                gemm->setAttr("weight_addr", builder.getI64IntegerAttr(wBase + weightOffset));
                gemm->setAttr("bias_addr",   builder.getI64IntegerAttr(bBase + biasOffset));

                SmallVector<double> bScales = arrF(gemm.b_scale());
                int64_t factor = computeConvFactor(
                    toDouble(gemm.a_scale()), bScales,
                    toDouble(gemm.out_scale()));
                gemm->setAttr("factor", builder.getI64IntegerAttr(factor));

                weightOffset += wSize;
                biasOffset   += bSize;
                isFirstOp = false;

            } else if (auto pool = dyn_cast<MaxPoolOp>(op)) {
                auto [inAddr, outAddr] = getActAddrs(isFirstOp);
                pool->setAttr("in_addr",  builder.getI64IntegerAttr(inAddr));
                pool->setAttr("out_addr", builder.getI64IntegerAttr(outAddr));
                isFirstOp = false;

            } else if (auto add = dyn_cast<QLinearAddOp>(op)) {
                // Both inputs are in the activation buffer region
                add->setAttr("in_addr",  builder.getI64IntegerAttr(actBase));
                add->setAttr("in2_addr", builder.getI64IntegerAttr(actBase));
                add->setAttr("out_addr", builder.getI64IntegerAttr(actBase));

                auto [f, f2] = computeAddFactors(
                    toDouble(add.a_scale()), toDouble(add.b_scale()),
                    toDouble(add.out_scale()));
                add->setAttr("factor",  builder.getI64IntegerAttr(f));
                add->setAttr("factor2", builder.getI64IntegerAttr(f2));
                isFirstOp = false;

            } else if (auto gap = dyn_cast<QLinearGlobalAveragePoolOp>(op)) {
                auto [inAddr, outAddr] = getActAddrs(isFirstOp);
                gap->setAttr("in_addr",  builder.getI64IntegerAttr(inAddr));
                gap->setAttr("out_addr", builder.getI64IntegerAttr(outAddr));
                isFirstOp = false;
            }
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOAAddrAssignPass() {
    return std::make_unique<COAAddrAssignPass>();
}

} // namespace mlir::coa
