//===- Tiling.cpp - COA hardware-aware tiling pass -----------------*- C++ -*-===//
//
// Pass: --coa-tiling
//
// Ports the Python get_tile() / calculate_buffer_consumption() logic from
// tools/assign_addr.py to a proper MLIR pass.  Writes tM / tR / tC attributes.
//
// Buffer constraints (configurable via pass options):
//   wdepth = ceil(tM/16) * ceil(tN/32) * kH * kW  < wdepthLimit (default 256)
//   gdepth = (tR-1)*s+(k-1)*d+1) * ((tC-1)*s+(k-1)*d+1) * ceil(tN/32)  < gdepthLimit (1024)
//   odepth = tR * tC * ceil(tM/16)                                        < odepthLimit (2048)
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <cstdint>

namespace mlir::coa {

#define GEN_PASS_CLASSES
#include "COA/COAPasses.h.inc"

namespace {

static int64_t arrI(ArrayAttr a, unsigned i, int64_t def = 0) {
    if (!a || i >= a.size()) return def;
    return a[i].cast<IntegerAttr>().getInt();
}
static int64_t optI(llvm::Optional<::mlir::ArrayAttr> a, unsigned i, int64_t def = 0) {
    if (!a) return def;
    return arrI(*a, i, def);
}

/// Ceiling division.
static int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// Buffer consumption check (returns 0=OK, 1=wdepth, 2=gdepth, 3=odepth over limit).
static int checkBuffers(int64_t tN, int64_t tM, int64_t tR, int64_t tC,
                        int64_t N, int64_t M, int64_t k, int64_t s, int64_t d,
                        int64_t wLimit, int64_t gLimit, int64_t oLimit) {
    // Maximum tN32 groups touched in a single tN-wide strip
    int64_t tN32_rd = 0;
    for (int64_t sn = 0; sn < N; sn += tN) {
        int64_t t = ceilDiv(sn + tN, 32) - sn / 32;
        tN32_rd = std::max(tN32_rd, t);
    }
    int64_t tM32_rd = 0;
    for (int64_t sm = 0; sm < M; sm += tM) {
        int64_t t = ceilDiv(sm + tM, 16) - sm / 16;
        tM32_rd = std::max(tM32_rd, t);
    }

    int64_t relems = (tR - 1) * s + (k - 1) * d + 1;
    int64_t celems = (tC - 1) * s + (k - 1) * d + 1;
    int64_t gdepth = relems * celems * tN32_rd;
    int64_t wdepth = tN32_rd * k * k * tM32_rd;
    int64_t odepth = tR * tC * tM32_rd;

    if (wdepth >= wLimit) return 1;
    if (gdepth >= gLimit) return 2;
    if (odepth >= oLimit) return 3;
    return 0;
}

/// Find largest legal tile (tM, tR, tC) using the same greedy heuristic as
/// the Python get_tile() in assign_addr.py.
static std::tuple<int64_t, int64_t, int64_t>
getTile(int64_t N, int64_t M, int64_t R, int64_t C,
        int64_t k, int64_t s, int64_t d,
        int64_t wLim, int64_t gLim, int64_t oLim) {
    int64_t tM = M, tR = R, tC = C;

    auto flag = [&]() {
        return checkBuffers(N, tM, tR, tC, N, M, k, s, d, wLim, gLim, oLim);
    };

    int f = flag();
    while (f != 0) {
        if (f == 1) {
            // wdepth over limit: reduce tM
            if      (tM % 2 == 0) tM /= 2;
            else if (tM % 3 == 0) tM /= 3;
            else if (tM % 5 == 0) tM /= 5;
            else                  tM = 16;
            if (tM < 16) tM = 16;
        } else {
            // gdepth or odepth over limit: reduce tR, then tC
            if (tR != 1) {
                if      (tR % 2 == 0) tR /= 2;
                else if (tR % 3 == 0) tR /= 3;
                else                  tR = 1;
            } else if (tC != 1) {
                if      (tC % 2 == 0) tC /= 2;
                else if (tC % 3 == 0) tC /= 3;
                else                  tC = 1;
            } else {
                break; // Can't reduce further
            }
        }
        if (tM < 16) tM = 16;
        f = flag();
    }

    // Try to greedily enlarge tR (try integer multiples)
    for (int64_t scale = 2; scale * tR <= R; ++scale) {
        int64_t newTR = scale * tR;
        if (checkBuffers(N, tM, newTR, tC, N, M, k, s, d, wLim, gLim, oLim) == 0)
            tR = newTR;
    }

    return {tM, tR, tC};
}

struct COATilingPass : public COATilingBase<COATilingPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        OpBuilder builder(funcOp.getContext());

        funcOp.walk([&](Operation *op) {
            builder.setInsertionPoint(op);
            int64_t wLim = wdepthLimit, gLim = gdepthLimit, oLim = odepthLimit;

            if (auto conv = dyn_cast<QLinearConvOp>(op)) {
                int64_t N = conv.N(), M = conv.M();
                int64_t R = conv.R(), C = conv.C();
                if (R == 0 || C == 0 || M == 0 || N == 0) return;

                auto kernel  = conv.kernel_shape();
                auto strides = conv.strides();
                auto dils    = conv.dilations();
                int64_t k = arrI(kernel,0,1), s = optI(strides,0,1), d = optI(dils,0,1);

                auto [tM, tR, tC] = getTile(N, M, R, C, k, s, d, wLim, gLim, oLim);
                conv->setAttr("tM", builder.getI64IntegerAttr(tM));
                conv->setAttr("tR", builder.getI64IntegerAttr(tR));
                conv->setAttr("tC", builder.getI64IntegerAttr(tC));

            } else if (auto gemm = dyn_cast<QGemmOp>(op)) {
                int64_t N = gemm.N(), M = gemm.M();
                if (M == 0 || N == 0) return;
                // GEMM treated as 1x1 convolution
                auto [tM, tR, tC] = getTile(N, M, 1, 1, 1, 1, 1, wLim, gLim, oLim);
                gemm->setAttr("tM", builder.getI64IntegerAttr(tM));
                gemm->setAttr("tR", builder.getI64IntegerAttr(1));
                gemm->setAttr("tC", builder.getI64IntegerAttr(1));

            } else if (auto pool = dyn_cast<MaxPoolOp>(op)) {
                int64_t R = pool.R(), C = pool.C(), M = pool.M(), N = pool.N();
                if (R == 0 || C == 0) return;
                auto kernel = pool.kernel_shape();
                auto strides = pool.strides();
                int64_t k = arrI(kernel,0,2), s = optI(strides,0,2);
                auto [tM, tR, tC] = getTile(N, M, R, C, k, s, 1, wLim, gLim, oLim);
                pool->setAttr("tM", builder.getI64IntegerAttr(tM));
                pool->setAttr("tR", builder.getI64IntegerAttr(tR));
                pool->setAttr("tC", builder.getI64IntegerAttr(tC));

            } else if (auto add = dyn_cast<QLinearAddOp>(op)) {
                int64_t R = add.R(), C = add.C(), M = add.M();
                if (R == 0 || C == 0 || M == 0) return;
                // For Add, tiling mirrors the spatial dimensions directly
                auto [tM, tR, tC] = getTile(M, M, R, C, 1, 1, 1, wLim, gLim, oLim);
                add->setAttr("tM", builder.getI64IntegerAttr(tM));
                add->setAttr("tR", builder.getI64IntegerAttr(tR));
                add->setAttr("tC", builder.getI64IntegerAttr(tC));
            }
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOATilingPass() {
    return std::make_unique<COATilingPass>();
}

} // namespace mlir::coa
