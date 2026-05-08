//===- Legalize.cpp - COA legalization / constraint checking pass --*- C++ -*-===//
//
// Pass: --coa-legalize
//
// Validates that all hardware attributes on COA ops satisfy the FPGA VLIW
// bit-field constraints before code generation is invoked.  Emits diagnostics
// and signals pass failure on any violation.
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::coa {

#define GEN_PASS_CLASSES
#include "COA/COAPasses.h.inc"

namespace {

/// Maximum values allowed by VLIW bit-field widths.
static constexpr int64_t kMaxDim11  = (1LL << 11) - 1;  // 2047  (R, C, R0, C0)
static constexpr int64_t kMaxDim12  = (1LL << 12) - 1;  // 4095  (M, N, sM, Mc)
static constexpr int64_t kMaxAddr36 = (1LL << 36) - 1;
static constexpr int64_t kMaxFactor34 = (1LL << 33) - 1; // signed 34-bit max

static bool check11(int64_t v) { return v >= 0 && v <= kMaxDim11; }
static bool check12(int64_t v) { return v >= 0 && v <= kMaxDim12; }
static bool checkAddr(int64_t v) { return v >= 0 && v <= kMaxAddr36; }
static bool checkFactor(int64_t v) {
    return v >= -(1LL << 33) && v <= kMaxFactor34;
}

struct COALegalizePass : public COALegalizeBase<COALegalizePass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        bool failed = false;

        auto checkDims = [&](Operation *op, int64_t R, int64_t C,
                             int64_t M, int64_t N, int64_t R0, int64_t C0) {
            if (!check11(R) || !check11(C) || !check11(R0) || !check11(C0)) {
                op->emitError("COA legalize: spatial dimensions exceed 11-bit limit (max 2047): "
                              "R=" + std::to_string(R) + " C=" + std::to_string(C));
                failed = true;
            }
            if (!check12(M) || !check12(N)) {
                op->emitError("COA legalize: channel dimensions exceed 12-bit limit (max 4095): "
                              "M=" + std::to_string(M) + " N=" + std::to_string(N));
                failed = true;
            }
        };

        auto checkAddrs = [&](Operation *op, std::initializer_list<int64_t> addrs) {
            for (int64_t a : addrs) {
                if (!checkAddr(a)) {
                    op->emitError("COA legalize: DDR address 0x" +
                                  std::to_string(static_cast<uint64_t>(a)) +
                                  " exceeds 36-bit field");
                    failed = true;
                }
            }
        };

        funcOp.walk([&](Operation *op) {
            if (auto conv = dyn_cast<QLinearConvOp>(op)) {
                checkDims(op, conv.R(), conv.C(), conv.M(), conv.N(),
                          conv.R0(), conv.C0());
                checkAddrs(op, {conv.in_addr(), conv.out_addr(),
                                conv.weight_addr(), conv.bias_addr()});
                if (conv.tM() < 16)
                    op->emitWarning("COA legalize: tM=" + std::to_string(conv.tM()) +
                                    " < 16 (minimum hardware granularity)");
                if (!checkFactor(conv.factor())) {
                    op->emitError("COA legalize: factor " + std::to_string(conv.factor()) +
                                  " overflows 34-bit signed field");
                    failed = true;
                }

            } else if (auto pool = dyn_cast<MaxPoolOp>(op)) {
                checkDims(op, pool.R(), pool.C(), pool.M(), pool.N(),
                          pool.R0(), pool.C0());
                checkAddrs(op, {pool.in_addr(), pool.out_addr()});

            } else if (auto add = dyn_cast<QLinearAddOp>(op)) {
                checkDims(op, add.R(), add.C(), add.M(), add.N(),
                          add.R0(), add.C0());
                checkAddrs(op, {add.in_addr(), add.in2_addr(), add.out_addr()});

            } else if (auto gap = dyn_cast<QLinearGlobalAveragePoolOp>(op)) {
                checkDims(op, gap.R(), gap.C(), gap.M(), gap.N(),
                          gap.R0(), gap.C0());
                checkAddrs(op, {gap.in_addr(), gap.out_addr()});

            } else if (auto gemm = dyn_cast<QGemmOp>(op)) {
                checkAddrs(op, {gemm.in_addr(), gemm.out_addr(),
                                gemm.weight_addr(), gemm.bias_addr()});
                if (!checkFactor(gemm.factor())) {
                    op->emitError("COA legalize: GEMM factor overflows 34-bit signed field");
                    failed = true;
                }
            }
        });

        if (failed)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOALegalizePass() {
    return std::make_unique<COALegalizePass>();
}

} // namespace mlir::coa
