//===- VLIWCodeGen.cpp - COA VLIW binary code generator -----------*- C++ -*-===//
//
// Pass: --coa-vliw-gen
//
// Translates hardware-lowered COA ops to 64-byte VLIW binary instructions.
// Ports tools/vliw.py (to_bytes() bit-packing) and extract_vliw.py field mapping.
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "COA/COAPasses.h"
#include "COA/VLIWDefs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <vector>

namespace mlir::coa {

#define GEN_PASS_DEF_COAVLIWGEN
#include "COA/COAPasses.h.inc"

//===----------------------------------------------------------------------===//
// VLIW::toBytes() implementation
// Bit-packs fields LSB-first into 64 bytes, exactly matching Python vliw.py.
//===----------------------------------------------------------------------===//

std::array<uint8_t, 64> VLIW::toBytes() const {
    std::array<uint8_t, 64> out{};
    out.fill(0);

    // Pairs of (value, num_bits) in field order - must match VLIWDefs.h layout.
    struct Field { uint64_t value; int bits; };
    std::vector<Field> fields = {
        {static_cast<uint64_t>(op),           8},
        {static_cast<uint64_t>(ddr_x1),      36},
        {static_cast<uint64_t>(ddr_x2),      36},
        {static_cast<uint64_t>(bias_addr),   11},
        {static_cast<uint64_t>(result_addr), 36},
        {static_cast<uint64_t>(lut_addr),     8},
        {static_cast<uint64_t>(R),           11},
        {static_cast<uint64_t>(C),           11},
        {static_cast<uint64_t>(M),           12},
        {static_cast<uint64_t>(N),           12},
        {static_cast<uint64_t>(R0),          11},
        {static_cast<uint64_t>(C0),          11},
        {static_cast<uint64_t>(sM_concat),   12},
        {static_cast<uint64_t>(M_concat),    12},
        {static_cast<uint64_t>(static_cast<uint8_t>(quant_x1_z)), 8},
        {static_cast<uint64_t>(static_cast<uint8_t>(quant_x2_z)), 8},
        {static_cast<uint64_t>(static_cast<uint8_t>(quant_y_z)),  8},
        {static_cast<uint64_t>(pad),          3},
        {static_cast<uint64_t>(kernel),       5},
        {static_cast<uint64_t>(stride),       3},
        {static_cast<uint64_t>(dilation),     3},
        {static_cast<uint64_t>(tR),          11},
        {static_cast<uint64_t>(tC),          11},
        {static_cast<uint64_t>(tM),          12},
        {static_cast<uint64_t>(tN),          12},
        {static_cast<uint64_t>(permuteR),     2},
        {static_cast<uint64_t>(permuteC),     2},
        {static_cast<uint64_t>(permuteM),     2},
        {static_cast<uint64_t>(permuteN),     2},
        {static_cast<uint64_t>(quant_factor),  34},
        {static_cast<uint64_t>(quant_factor2), 32},
        {0ULL,                               127},  // reserved
    };

    int bitPos = 0;
    for (auto &f : fields) {
        for (int i = 0; i < f.bits; ++i) {
            int byteIdx = bitPos / 8;
            int bitIdx  = bitPos % 8;
            if (byteIdx < 64) {
                uint8_t bit = (f.value >> i) & 1u;
                out[byteIdx] |= static_cast<uint8_t>(bit << bitIdx);
            }
            ++bitPos;
        }
    }
    return out;
}

std::string VLIW::repr() const {
    return "VLIW(op=" + std::to_string(op) +
           " ddr_x1=0x" + llvm::utohexstr(ddr_x1) +
           " R=" + std::to_string(R) + " C=" + std::to_string(C) +
           " M=" + std::to_string(M) + " N=" + std::to_string(N) + ")";
}

std::vector<uint8_t> packVLIWs(const std::vector<VLIW> &instrs) {
    std::vector<uint8_t> buf;
    buf.reserve(instrs.size() * 64);
    for (const auto &v : instrs) {
        auto b = v.toBytes();
        buf.insert(buf.end(), b.begin(), b.end());
    }
    return buf;
}

//===----------------------------------------------------------------------===//
// MLIR Pass
//===----------------------------------------------------------------------===//

namespace {

/// Build a VLIW from a qlinearconv op.
static VLIW buildConvVLIW(QLinearConvOp conv) {
    VLIW v;
    v.op          = static_cast<uint8_t>(VLIWOperator::Conv);
    v.ddr_x1      = static_cast<uint64_t>(conv.getInAddr());
    v.ddr_x2      = static_cast<uint64_t>(conv.getWeightAddr());
    v.bias_addr   = static_cast<uint32_t>(conv.getBiasAddr() & 0x7FF);
    v.result_addr = static_cast<uint64_t>(conv.getOutAddr());
    v.lut_addr    = static_cast<uint8_t>((conv.getSiluAddr() >> 24) & 0xFF);
    v.R  = static_cast<uint16_t>(conv.getR());
    v.C  = static_cast<uint16_t>(conv.getC());
    v.M  = static_cast<uint16_t>(conv.getM());
    v.N  = static_cast<uint16_t>(conv.getN());
    v.R0 = static_cast<uint16_t>(conv.getR0());
    v.C0 = static_cast<uint16_t>(conv.getC0());
    v.sM_concat = static_cast<uint16_t>(conv.getSMConcat());
    v.M_concat  = static_cast<uint16_t>(conv.getMConcat());
    v.quant_x1_z = static_cast<int8_t>(conv.getInZp());
    v.quant_x2_z = 0; // per-channel weight ZP handled by factor
    v.quant_y_z  = static_cast<int8_t>(conv.getOutZp());
    auto pads = conv.getPads();
    auto k    = conv.getKernelShape();
    auto s    = conv.getStrides();
    auto d    = conv.getDilations();
    v.pad      = static_cast<uint8_t>(pads[0]);
    v.kernel   = static_cast<uint8_t>(k[0]);
    v.stride   = static_cast<uint8_t>(s[0]);
    v.dilation = static_cast<uint8_t>(d[0]);
    v.tR = static_cast<uint16_t>(conv.getTR());
    v.tC = static_cast<uint16_t>(conv.getTC());
    v.tM = static_cast<uint16_t>(conv.getTM());
    v.tN = static_cast<uint16_t>(conv.getN());
    v.quant_factor = conv.getFactor();
    return v;
}

static VLIW buildPoolVLIW(MaxPoolOp pool) {
    VLIW v;
    v.op          = static_cast<uint8_t>(VLIWOperator::Pool);
    v.ddr_x1      = static_cast<uint64_t>(pool.getInAddr());
    v.result_addr = static_cast<uint64_t>(pool.getOutAddr());
    v.R  = static_cast<uint16_t>(pool.getR());
    v.C  = static_cast<uint16_t>(pool.getC());
    v.M  = static_cast<uint16_t>(pool.getM());
    v.N  = static_cast<uint16_t>(pool.getN());
    v.R0 = static_cast<uint16_t>(pool.getR0());
    v.C0 = static_cast<uint16_t>(pool.getC0());
    auto k = pool.getKernelShape();
    auto s = pool.getStrides();
    auto p = pool.getPads();
    v.kernel = static_cast<uint8_t>(k[0]);
    v.stride = static_cast<uint8_t>(s[0]);
    v.pad    = static_cast<uint8_t>(p[0]);
    v.tR = static_cast<uint16_t>(pool.getTR());
    v.tC = static_cast<uint16_t>(pool.getTC());
    v.tM = static_cast<uint16_t>(pool.getTM());
    v.tN = static_cast<uint16_t>(pool.getN());
    return v;
}

static VLIW buildAddVLIW(QLinearAddOp add) {
    VLIW v;
    v.op          = static_cast<uint8_t>(VLIWOperator::Add);
    v.ddr_x1      = static_cast<uint64_t>(add.getInAddr());
    v.ddr_x2      = static_cast<uint64_t>(add.getIn2Addr());
    v.result_addr = static_cast<uint64_t>(add.getOutAddr());
    v.R  = static_cast<uint16_t>(add.getR());
    v.C  = static_cast<uint16_t>(add.getC());
    v.M  = static_cast<uint16_t>(add.getM());
    v.N  = static_cast<uint16_t>(add.getN());
    v.R0 = static_cast<uint16_t>(add.getR0());
    v.C0 = static_cast<uint16_t>(add.getC0());
    v.quant_x1_z  = static_cast<int8_t>(add.getAZp());
    v.quant_x2_z  = static_cast<int8_t>(add.getBZp());
    v.quant_y_z   = static_cast<int8_t>(add.getOutZp());
    v.quant_factor  = add.getFactor();
    v.quant_factor2 = add.getFactor2();
    return v;
}

static VLIW buildGAPVLIW(QLinearGlobalAveragePoolOp gap) {
    VLIW v;
    v.op          = static_cast<uint8_t>(VLIWOperator::GAP);
    v.ddr_x1      = static_cast<uint64_t>(gap.getInAddr());
    v.result_addr = static_cast<uint64_t>(gap.getOutAddr());
    v.R  = 1;
    v.C  = 1;
    v.M  = static_cast<uint16_t>(gap.getM());
    v.N  = static_cast<uint16_t>(gap.getN());
    v.R0 = static_cast<uint16_t>(gap.getR0());
    v.C0 = static_cast<uint16_t>(gap.getC0());
    v.quant_x1_z = static_cast<int8_t>(gap.getInZp());
    v.quant_y_z  = static_cast<int8_t>(gap.getOutZp());
    return v;
}

static VLIW buildGemmVLIW(QGemmOp gemm) {
    VLIW v;
    v.op          = static_cast<uint8_t>(VLIWOperator::Conv); // GEMM = Conv type
    v.ddr_x1      = static_cast<uint64_t>(gemm.getInAddr());
    v.ddr_x2      = static_cast<uint64_t>(gemm.getWeightAddr());
    v.bias_addr   = static_cast<uint32_t>(gemm.getBiasAddr() & 0x7FF);
    v.result_addr = static_cast<uint64_t>(gemm.getOutAddr());
    v.R  = 1; v.C = 1; v.R0 = 1; v.C0 = 1;
    v.M  = static_cast<uint16_t>(gemm.getM());
    v.N  = static_cast<uint16_t>(gemm.getN());
    v.sM_concat = static_cast<uint16_t>(gemm.getSMConcat());
    v.M_concat  = static_cast<uint16_t>(gemm.getMConcat());
    v.quant_x1_z = static_cast<int8_t>(gemm.getAZp());
    v.quant_y_z  = static_cast<int8_t>(gemm.getOutZp());
    v.kernel = 1; v.stride = 1; v.dilation = 1; v.pad = 0;
    v.tM = static_cast<uint16_t>(gemm.getTM());
    v.tR = 1; v.tC = 1;
    v.tN = static_cast<uint16_t>(gemm.getN());
    v.quant_factor = gemm.getFactor();
    return v;
}

struct COAVLIWGenPass : public impl::COAVLIWGenBase<COAVLIWGenPass> {
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        std::vector<VLIW> instrs;

        funcOp.walk([&](Operation *op) {
            if (auto conv = dyn_cast<QLinearConvOp>(op))
                instrs.push_back(buildConvVLIW(conv));
            else if (auto pool = dyn_cast<MaxPoolOp>(op))
                instrs.push_back(buildPoolVLIW(pool));
            else if (auto add = dyn_cast<QLinearAddOp>(op))
                instrs.push_back(buildAddVLIW(add));
            else if (auto gap = dyn_cast<QLinearGlobalAveragePoolOp>(op))
                instrs.push_back(buildGAPVLIW(gap));
            else if (auto gemm = dyn_cast<QGemmOp>(op))
                instrs.push_back(buildGemmVLIW(gemm));
        });

        // Write binary output
        auto binary = packVLIWs(instrs);
        std::ofstream ofs(outputFile, std::ios::binary);
        if (!ofs) {
            funcOp.emitError("coa-vliw-gen: cannot open output file: " + outputFile);
            signalPassFailure();
            return;
        }
        ofs.write(reinterpret_cast<const char *>(binary.data()),
                  static_cast<std::streamsize>(binary.size()));

        llvm::outs() << "[coa-vliw-gen] Wrote " << instrs.size()
                     << " VLIW instructions (" << binary.size()
                     << " bytes) to " << outputFile << "\n";
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createCOAVLIWGenPass() {
    return std::make_unique<COAVLIWGenPass>();
}

} // namespace mlir::coa
