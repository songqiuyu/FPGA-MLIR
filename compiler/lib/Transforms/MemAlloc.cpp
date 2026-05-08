//===- MemAlloc.cpp - COA activation memory allocation pass --------*- C++ -*-===//
//
// Pass: --coa-mem-alloc
//
// Implements liveness-based linear scan allocation for FPGA DDR activation
// tensors, replacing the broken ping-pong scheme in --coa-addr-assign.
//
// Algorithm (Poletto & Sarkar, TOPLAS 1999 - adapted for FPGA DDR):
//   1. Topological walk to assign sequence numbers to all COA ops.
//   2. Liveness analysis: for each op result (activation tensor), compute
//      [def_time, last_use_time, size_bytes].
//   3. Linear scan allocation over a DDR pool starting at act-base:
//        - On each interval start, expire dead intervals → free their blocks.
//        - Best-fit from free pool; fall back to bump allocation.
//   4. Write in_addr / out_addr / in2_addr attributes on every COA op.
//
// Leaves weight_addr, bias_addr, factor, factor2 to --coa-addr-assign.
//
// Academic note:
//   Ping-Pong uses 2 × max_tensor_size regardless of actual lifetimes.
//   Linear scan uses peak(Σ simultaneously-live sizes), which is optimal
//   for the static schedule produced by the COA compiler.
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace mlir::coa {

#define GEN_PASS_CLASSES
#include "COA/COAPasses.h.inc"

namespace {

/// Alignment for every activation allocation (bytes). Matches typical DMA
/// burst size on FPGA AXI buses.
static constexpr int64_t kActAlign = 64;

/// Round x up to the nearest multiple of align.
static int64_t alignTo(int64_t x, int64_t align) {
    return ((x + align - 1) / align) * align;
}

/// Return the output activation tensor size in bytes for a COA op.
/// Reads M, R, C attributes populated by --coa-shape-infer.
///   Conv / Pool / Add: size = M × R × C (INT8, 1 byte per element)
///   GAP / GEMM:        R = C = 1, so size = M
static int64_t getActSize(Operation *op) {
    auto getI = [&](StringRef name) -> int64_t {
        if (auto a = op->getAttrOfType<IntegerAttr>(name)) return a.getInt();
        return 0;
    };
    int64_t M = getI("M"), R = getI("R"), C = getI("C");
    if (M <= 0) return kActAlign; // safe fallback
    int64_t sz = M;
    if (R > 1) sz *= R;
    if (C > 1) sz *= C;
    return sz;
}

/// Liveness interval: [defTime, lastUse] for one activation SSA value.
struct LiveInterval {
    int64_t defTime;      ///< Sequence number of the defining op.
    int64_t lastUse;      ///< Sequence number of the last consuming op.
    int64_t size;         ///< Aligned allocation size in bytes.
    int64_t assignedAddr; ///< DDR address assigned by linear scan.
    mlir::Value value;    ///< The MLIR SSA value (op result).
};

/// An available block in the DDR activation pool.
struct FreeBlock {
    int64_t addr;
    int64_t size;
};

// ─────────────────────────────────────────────────────────────────────────────

struct COAMemAllocPass : public COAMemAllocBase<COAMemAllocPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        OpBuilder builder(funcOp.getContext());

        // ── Phase 1: Assign monotone sequence numbers to all COA ops. ─────────
        llvm::DenseMap<Operation *, int64_t> seqMap;
        int64_t seqNo = 0;
        funcOp.walk([&](Operation *op) {
            if (isa<QLinearConvOp, MaxPoolOp, QLinearAddOp,
                    QLinearGlobalAveragePoolOp, QGemmOp>(op))
                seqMap[op] = seqNo++;
        });
        int64_t numOps = seqNo;

        // ── Phase 2: Build liveness intervals. ───────────────────────────────
        // The network input is funcArg[0]; it lives at DDR address 0x0.
        llvm::DenseMap<mlir::Value, int64_t> addrMap;
        if (funcOp.getNumArguments() > 0)
            addrMap[funcOp.getArgument(0)] = 0LL;

        std::vector<LiveInterval> intervals;
        intervals.reserve(numOps);

        funcOp.walk([&](Operation *op) {
            if (!isa<QLinearConvOp, MaxPoolOp, QLinearAddOp,
                       QLinearGlobalAveragePoolOp, QGemmOp>(op))
                return;

            int64_t defT = seqMap[op];
            int64_t sz   = alignTo(getActSize(op), kActAlign);

            // last_use = max sequence number among all consuming ops.
            // If a use belongs to a non-COA op (e.g., func.return), treat it
            // as live until the end of the program.
            int64_t lastT = defT;
            for (auto &use : op->getResult(0).getUses()) {
                Operation *user = use.getOwner();
                auto it = seqMap.find(user);
                lastT = std::max(lastT, (it != seqMap.end()) ? it->second : numOps);
            }

            intervals.push_back({defT, lastT, sz, 0LL, op->getResult(0)});
        });

        // ── Phase 3: Linear Scan Allocation. ─────────────────────────────────
        // Sort intervals by definition time.
        std::sort(intervals.begin(), intervals.end(),
                  [](const LiveInterval &a, const LiveInterval &b) {
                      return a.defTime < b.defTime;
                  });

        int64_t allocPtr    = activationBase; // bump pointer (hwm)
        int64_t peakUsage   = activationBase;

        // active[i] = index into `intervals`, sorted by lastUse ascending.
        std::vector<size_t> active;
        std::vector<FreeBlock> freePool;

        // Expire intervals that ended before curTime and reclaim their memory.
        auto expireActive = [&](int64_t curTime) {
            std::vector<size_t> stillLive;
            for (size_t idx : active) {
                if (intervals[idx].lastUse < curTime) {
                    freePool.push_back(
                        {intervals[idx].assignedAddr, intervals[idx].size});
                } else {
                    stillLive.push_back(idx);
                }
            }
            active = std::move(stillLive);
        };

        // Best-fit: choose the smallest free block that is ≥ needed bytes.
        // Returns the allocated address, or -1 if no suitable block found.
        auto bestFitAlloc = [&](int64_t needed) -> int64_t {
            int bestIdx  = -1;
            int64_t bestSz = INT64_MAX;
            for (int i = 0; i < (int)freePool.size(); ++i) {
                if (freePool[i].size >= needed && freePool[i].size < bestSz) {
                    bestSz  = freePool[i].size;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) return -1LL;

            int64_t addr = freePool[bestIdx].addr;
            int64_t rem  = freePool[bestIdx].size - needed;
            if (rem >= kActAlign) {
                // Return the unused tail to the free pool.
                freePool[bestIdx] = {addr + needed, rem};
            } else {
                freePool.erase(freePool.begin() + bestIdx);
            }
            return addr;
        };

        for (size_t i = 0; i < intervals.size(); ++i) {
            LiveInterval &iv = intervals[i];
            expireActive(iv.defTime);

            int64_t addr = bestFitAlloc(iv.size);
            if (addr < 0) {
                // No reusable block; bump-allocate from the high-water mark.
                addr      = allocPtr;
                allocPtr += iv.size;
                peakUsage = std::max(peakUsage, allocPtr);
            }
            iv.assignedAddr = addr;
            addrMap[iv.value] = addr;

            // Insert into active, keeping it sorted by lastUse.
            auto pos = std::lower_bound(
                active.begin(), active.end(), i,
                [&](size_t a, size_t b) {
                    return intervals[a].lastUse < intervals[b].lastUse;
                });
            active.insert(pos, i);
        }

        // ── Reporting. ────────────────────────────────────────────────────────
        int64_t peakBytes = peakUsage - activationBase;

        // Ping-pong baseline: 2 × maximum single activation tensor size.
        int64_t maxSingle = 0;
        for (auto &iv : intervals)
            maxSingle = std::max(maxSingle, iv.size);
        int64_t pingPongBytes = 2 * maxSingle;

        if (verbose) {
            llvm::outs() << "[coa-mem-alloc] Liveness + Linear Scan DDR Allocation:\n";
            for (size_t i = 0; i < intervals.size(); ++i) {
                auto &iv = intervals[i];
                llvm::outs() << llvm::format(
                    "  interval[%2zu]  t=[%2lld,%2lld]  size=%7lld B  "
                    "addr=0x%08llx\n",
                    i, (long long)iv.defTime, (long long)iv.lastUse,
                    (long long)iv.size, (long long)iv.assignedAddr);
            }
            llvm::outs() << llvm::format(
                "  Peak footprint : %lld bytes\n"
                "  Ping-Pong base : %lld bytes",
                (long long)peakBytes, (long long)pingPongBytes);
            if (pingPongBytes > 0) {
                int pct = (int)(100LL * (pingPongBytes - peakBytes) / pingPongBytes);
                llvm::outs() << llvm::format("  (saved %d%%)", pct);
            }
            llvm::outs() << "\n";
        } else {
            llvm::outs() << llvm::format(
                "[coa-mem-alloc] Peak DDR activation: %lld B  "
                "(Ping-Pong baseline: %lld B)\n",
                (long long)peakBytes, (long long)pingPongBytes);
        }

        // ── Phase 4: Write in_addr / out_addr / in2_addr. ─────────────────────
        auto getAddr = [&](mlir::Value v) -> int64_t {
            auto it = addrMap.find(v);
            return (it != addrMap.end()) ? it->second : 0LL;
        };

        funcOp.walk([&](Operation *op) {
            builder.setInsertionPoint(op);
            if (isa<QLinearConvOp>(op)) {
                // operand[0] = input activation
                op->setAttr("in_addr",
                    builder.getI64IntegerAttr(getAddr(op->getOperand(0))));
                op->setAttr("out_addr",
                    builder.getI64IntegerAttr(getAddr(op->getResult(0))));

            } else if (isa<QGemmOp>(op)) {
                // operand[0] = A (activation input)
                op->setAttr("in_addr",
                    builder.getI64IntegerAttr(getAddr(op->getOperand(0))));
                op->setAttr("out_addr",
                    builder.getI64IntegerAttr(getAddr(op->getResult(0))));

            } else if (isa<MaxPoolOp>(op)) {
                // operand[0] = input activation
                op->setAttr("in_addr",
                    builder.getI64IntegerAttr(getAddr(op->getOperand(0))));
                op->setAttr("out_addr",
                    builder.getI64IntegerAttr(getAddr(op->getResult(0))));

            } else if (isa<QLinearAddOp>(op)) {
                // operand[0] = a (main path)
                // operand[1] = b (skip connection / shortcut)
                op->setAttr("in_addr",
                    builder.getI64IntegerAttr(getAddr(op->getOperand(0))));
                op->setAttr("in2_addr",
                    builder.getI64IntegerAttr(getAddr(op->getOperand(1))));
                op->setAttr("out_addr",
                    builder.getI64IntegerAttr(getAddr(op->getResult(0))));

            } else if (isa<QLinearGlobalAveragePoolOp>(op)) {
                // operand[0] = input activation
                op->setAttr("in_addr",
                    builder.getI64IntegerAttr(getAddr(op->getOperand(0))));
                op->setAttr("out_addr",
                    builder.getI64IntegerAttr(getAddr(op->getResult(0))));
            }
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOAMemAllocPass() {
    return std::make_unique<COAMemAllocPass>();
}

} // namespace mlir::coa
