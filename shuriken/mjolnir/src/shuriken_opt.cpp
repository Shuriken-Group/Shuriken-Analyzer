#include "shuriken_opt.h"
// Dex & APK stuff
#include "transform/lifter.h"
#include "passes/mjolnirtoopgraph.h"
#include "mjolnir/MjolnIROps.h"
#include <algorithm>
#include <cassert>
#include <cstdio>

#include "fmt/color.h"
#include "fmt/core.h"
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/ViewOpGraph.h>

#include <string>
#include <vector>

void shuriken_opt_log(const std::string &msg);
mlir::LogicalResult generate_functions_cfg(mlir::raw_ostream &os, mlir::OwningOpRef<mlir::ModuleOp> &module);

bool LOGGING = false;

int main(int argc, char **argv) {
    // list of arguments
    std::vector<std::string> args{argv, argv + argc};

    std::map<std::string, std::string> options{
            {"-f", ""},
            {"--file", ""}
    };


    // INFO: Check if we need to print out help
    bool need_help = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-h" || arg == "--help"; }) != args.end();
    LOGGING = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-d" || arg == "--diagnostics"; }) != args.end();
    bool show_graph = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-g" || arg == "--graph"; }) != args.end();
    if (LOGGING)
        shuriken_opt_log(fmt::format("\nLOGGING IS ENABLED\n"));

    // INFO: Error is true if something is wrong with input
    bool error = acquire_input(args, options);


    if (need_help || error) {
        show_help(args[0]);
    }


    if (error)
        return -1;


    /// INFO: Process a file.
    if (options.at("-f") != "" || options.at("--file") != "") {
        std::string file_name = options.at("-f") != "" ? options.at("-f") : options.at("--file");
        shuriken_opt_log(fmt::format("The file name is {}\n", file_name));
        shuriken::MjolnIR::Lifter lifter(file_name, false, true);

        for (auto &module: lifter.mlir_gen_result) { module->dump(); }

        if (show_graph) {

            for (auto &module : lifter.mlir_gen_result) {
                auto logical_result = generate_functions_cfg(llvm::outs(), module);
                if (logical_result.failed())
                    return -1;
            }
        }

    }
    return 0;
}

void show_help(std::string &prog_name) {
    fmt::print(fg(fmt::color::green), "USAGE: {} [-h | --help] [-d | --diagnostics] [-f|--file file_name] \n", prog_name);
    fmt::print(fg(fmt::color::green), "    -h | --help: Shows the help menu, like what you're seeing right now\n");
    fmt::print(fg(fmt::color::green), "    -d | --diagnostics: Enables diagnostics for shuriken-opt\n");
    fmt::print(fg(fmt::color::green), "    -f | --file: Analyzes a file with file name\n");
    fmt::print(fg(fmt::color::green), "    -g | --graph: Dump the graph in dot format (needs a file)\n");
}
/// Simple log for shuriken opt, msgs need to provide newline.
void shuriken_opt_log(const std::string &msg) {
    if (LOGGING)
        fmt::print(stderr, "{}", msg);
}


mlir::LogicalResult generate_functions_cfg(mlir::raw_ostream &os, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    mlir::PassManager pm(module.get()->getName());

    pm.addNestedPass<MethodOp>(shuriken::MjolnIR::create_mjolnir_op_graph_pass(os));

    return pm.run(*module);
}