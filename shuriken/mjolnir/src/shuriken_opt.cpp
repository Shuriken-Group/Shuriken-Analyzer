#include "shuriken_opt.h"
// Dex & APK stuff
#include "passes/mjolnirtoopgraph.h"
#include "passes/opt.h"
#include "transform/lifter.h"
#include <algorithm>
#include <cstdio>

#include "fmt/color.h"
#include "fmt/core.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>

#include <string>
#include <vector>

void shuriken_opt_log(const std::string &msg);

bool LOGGING = false;

int main(int argc, char **argv) {
    // list of arguments
    std::vector<std::string> args{argv, argv + argc};

    std::map<std::string, std::string> options{
            {"-f", ""},
            {"--file", ""}};


    LOGGING = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-d" || arg == "--diagnostics"; }) != args.end();
    // INFO: Check if we need to print out help
    bool need_help = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-h" || arg == "--help"; }) != args.end();
    bool show_graph = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-g" || arg == "--graph"; }) != args.end();
    bool lift = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "--lift"; }) != args.end();
    bool lower = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "--lower"; }) != args.end();
    bool opt = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "--opt"; }) != args.end();

    if (LOGGING)
        shuriken_opt_log(fmt::format("\nLOGGING IS ENABLED\n"));

    // INFO: Lowering or optimization requires lifting
    if (lower || opt) lift = true;

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
        if (lift) {
            std::cerr << "*****BEGIN LIFTING*****\n";
            shuriken::MjolnIR::Lifter lifter(file_name, false, true);
            for (auto &module: lifter.mlir_gen_result) { module->dump(); }

            if (show_graph) {

                for (auto &module: lifter.mlir_gen_result) {
                    auto logical_result = shuriken::MjolnIR::generate_functions_cfg(llvm::outs(), module);
                    if (logical_result.failed())
                        return -1;
                }
            }
            if (opt) {
                std::cerr << "*****BEGIN Optimizer*****\n";
                for (auto &module: lifter.mlir_gen_result) {
                    auto result = shuriken::MjolnIR::Opt::run(module.get());
                    if (mlir::failed(result)) {
                        llvm::errs() << "Failed to optimize" << module.get().getName() << "\n";
                    } else {
                        module->dump();
                    }
                }
            }

        } else {
            std::cerr << "USer picked the -f | --file option but did not specify at least lift | lower | opt\n";
        }

        return 0;
    }

    if (need_help) return 0;
    else
        return -1;
}

void show_help(std::string &prog_name) {
    fmt::print(fg(fmt::color::green), "USAGE: {} [-h | --help] [-d | --diagnostics] [-f|--file file_name --lift --lower --opt] \n", prog_name);
    fmt::print(fg(fmt::color::green), "    -h | --help: Shows the help menu, like what you're seeing right now\n");
    fmt::print(fg(fmt::color::green), "    -d | --diagnostics: Enables diagnostics for shuriken-opt\n");
    fmt::print(fg(fmt::color::green), "    -f | --file: Analyzes a file with file name, needs additional information\n");
    fmt::print(fg(fmt::color::green), "    -lift : Lift the dex format up to MjolnIR\n");
    fmt::print(fg(fmt::color::green), "    -lower: Lower the lifted MjolnIR down to smali (Enables lift when this is opted)\n");
    fmt::print(fg(fmt::color::green), "    -opt: Run some default optimization (Nop removal for now)\n");
    fmt::print(fg(fmt::color::green), "    -g | --graph: Dump the graph in dot format (needs a file)\n");
}
/// Simple log for shuriken opt, msgs need to provide newline.
void shuriken_opt_log(const std::string &msg) {
    if (LOGGING)
        fmt::print(stderr, "{}", msg);
}
