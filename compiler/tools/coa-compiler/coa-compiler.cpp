//===- coa-compiler.cpp - End-to-end COA compiler driver ----------*- C++ -*-===//
//
// One-shot compiler: COA MLIR → full pass pipeline → VLIW binary.
//
// Usage:
//   coa-compiler [options] input.mlir
//
// Options:
//   --output <file>           Output VLIW binary (default: output.vliw)
//   --wdepth-limit <n>        Weight buffer limit (default: 256)
//   --gdepth-limit <n>        Input tile buffer limit (default: 1024)
//   --odepth-limit <n>        Output tile buffer limit (default: 2048)
//   --weight-base <hex>       DDR base for weights (default: 0x8000000)
//   --bias-base <hex>         DDR base for bias (default: 0xC0000000)
//   --act-base <hex>          DDR base for activations (default: 0x10000000)
//   --skip-legalize           Skip legalization check (for debugging)
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> InputFilename(
    cl::Positional, cl::desc("<input COA MLIR file>"), cl::Required);

static cl::opt<std::string> OutputFilename(
    "output", cl::desc("Output VLIW binary file"), cl::value_desc("file"),
    cl::init("output.vliw"));

static cl::opt<bool> SkipLegalize(
    "skip-legalize", cl::desc("Skip legalization pass"), cl::init(false));

static cl::opt<int64_t> WdepthLimit(
    "wdepth-limit", cl::desc("Weight buffer depth limit"), cl::init(256));
static cl::opt<int64_t> GdepthLimit(
    "gdepth-limit", cl::desc("Input tile buffer depth limit"), cl::init(1024));
static cl::opt<int64_t> OdepthLimit(
    "odepth-limit", cl::desc("Output tile buffer depth limit"), cl::init(2048));

static cl::opt<int64_t> WeightBase(
    "weight-base", cl::desc("DDR base address for weights"), cl::init(0x8000000));
static cl::opt<int64_t> BiasBase(
    "bias-base", cl::desc("DDR base address for bias"), cl::init((int64_t)0xC0000000));
static cl::opt<int64_t> ActBase(
    "act-base", cl::desc("DDR base address for activations"), cl::init(0x10000000));

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    cl::ParseCommandLineOptions(argc, argv, "COA MLIR to VLIW binary compiler\n");

    // --- Set up MLIR context ---
    MLIRContext context;
    DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<coa::COADialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    coa::registerCOAPasses();

    // --- Parse input MLIR ---
    std::string errorMessage;
    auto srcMgr = std::make_shared<llvm::SourceMgr>();
    auto inputFile = openInputFile(InputFilename, &errorMessage);
    if (!inputFile) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }
    srcMgr->AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

    ParserConfig parseConfig(&context);
    OwningOpRef<ModuleOp> module =
        parseSourceFile<ModuleOp>(*srcMgr, parseConfig);
    if (!module) {
        llvm::errs() << "coa-compiler: Failed to parse input MLIR\n";
        return 1;
    }

    // --- Build pass pipeline ---
    PassManager pm(&context);
    pm.enableVerifier(true);

    // Build the standard COA pipeline on each FuncOp
    auto &funcPM = pm.nest<func::FuncOp>();
    funcPM.addPass(coa::createCOAShapeInferPass());
    funcPM.addPass(coa::createCOAOpFusionPass());

    // Tiling with configurable limits (use pipeline string for MLIR 15 options)
    {
        std::string spec = "coa-tiling{wdepth-limit=" + std::to_string((int64_t)WdepthLimit)
                         + " gdepth-limit=" + std::to_string((int64_t)GdepthLimit)
                         + " odepth-limit=" + std::to_string((int64_t)OdepthLimit) + "}";
        if (failed(mlir::parsePassPipeline(spec, funcPM, llvm::errs())))
            return 1;
    }

    // Address assignment with configurable bases
    {
        std::string spec = "coa-addr-assign{weight-base=" + std::to_string((int64_t)WeightBase)
                         + " bias-base=" + std::to_string((int64_t)BiasBase)
                         + " act-base=" + std::to_string((int64_t)ActBase) + "}";
        if (failed(mlir::parsePassPipeline(spec, funcPM, llvm::errs())))
            return 1;
    }

    if (!SkipLegalize)
        funcPM.addPass(coa::createCOALegalizePass());

    // VLIW generation
    {
        std::string spec = "coa-vliw-gen{output=" + std::string(OutputFilename) + "}";
        if (failed(mlir::parsePassPipeline(spec, funcPM, llvm::errs())))
            return 1;
    }

    // --- Run the pipeline ---
    if (failed(pm.run(*module))) {
        llvm::errs() << "coa-compiler: Compilation failed.\n";
        return 1;
    }

    llvm::outs() << "coa-compiler: Done. Output written to " << OutputFilename << "\n";
    return 0;
}
