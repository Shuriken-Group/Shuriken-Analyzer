#include "mjolnir/MjolnIRDialect.h"
#include "shuriken/analysis/Dex/dex_analysis.h"
#include "shuriken_opt.h"
// Dex & APK stuff
#include "transform/lifter.h"
#include <cstdio>
#include <functional>
#include <memory>

#include "fmt/color.h"
#include <mlir/IR/MLIRContext.h>
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>
#include <string>
#include <variant>
#include <vector>
bool acquire_input(std::vector<std::string> &args, std::map<std::string, std::string> &options) {
    bool error = false;

    // INFO: Acquiring input, set error if input to option is not given.
    for (size_t i = 1; i < args.size(); i++) {
        auto &s = args[i];
        if (auto it = options.find(s); it != options.end()) {
            if (i + 1 < args.size()) {
                it->second = args[i + 1];
            } else {
                fmt::print(fg(fmt::color::red),
                           "ERROR: Provide input for {}\n", it->first);
                error = true;
            }
        }
    }
    return error;
}
auto getAnalysis(const std::string &file_name) -> std::variant<std::unique_ptr<AnalysisClass>, OptError> {

    auto parsed_dex = shuriken::parser::parse_dex(file_name);
    assert(parsed_dex);
    auto disassembler = std::make_unique<shuriken::disassembler::dex::DexDisassembler>(parsed_dex.get());
    assert(disassembler);
    disassembler->disassembly_dex();

    // INFO: xrefs option disabled
    auto analysis = std::make_unique<shuriken::analysis::dex::Analysis>(parsed_dex.get(), disassembler.get(), false);
    if (analysis) return analysis;
    return OptError::GenericError;
}


auto lift_ir(std::reference_wrapper<MethodAnalysis> mm, const bool LOGGING) -> void {
    mlir::DialectRegistry registry;
    registry.insert<::mlir::shuriken::MjolnIR::MjolnIRDialect>();

    mlir::MLIRContext context;
    context.loadAllAvailableDialects();

    shuriken::MjolnIR::Lifter lifter(context, false, LOGGING);

    auto module_op = lifter.mlirGen(&mm.get());

    module_op->dump();
}
