#include "shuriken_opt.h"
// Dex & APK stuff
#include "transform/lifter.h"
#include <algorithm>
#include <cstdio>
#include <memory>

#include "fmt/color.h"
#include "fmt/core.h"
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>
#include <string>
#include <variant>
#include <vector>
void shuriken_opt_log(const std::string &msg);
bool LOGGING = false;
int main(int argc, char **argv) {
    // list of arguments
    std::vector<std::string> args{argv, argv + argc};

    std::map<std::string, std::string> options{
            {"-f", ""},
            {"--file", ""},
    };


    // INFO: Check if we need to print out help
    bool need_help = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-h" || arg == "--help"; }) != args.end();
    LOGGING = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-d" || arg == "--diagnostics"; }) != args.end();
    if (LOGGING)
        shuriken_opt_log(fmt::format("\nLOGGING IS ENABLED\n"));

    // INFO: Error is true if something is wrong with input
    bool error = acquire_input(args, options);


    if (need_help || error) {
        show_help(args[0]);
    }


    if (error)
        return -1;

    if (options.at("-f") != "" || options.at("--file") != "") {
        std::string file_name = options.at("-f") != "" ? options.at("-f") : options.at("--file");
        shuriken_opt_log(fmt::format("The file name is {}\n", file_name));
        auto parsed_dex = shuriken::parser::parse_dex(file_name);
        assert(parsed_dex);
        auto disassembler = std::make_unique<shuriken::disassembler::dex::DexDisassembler>(parsed_dex.get());
        assert(disassembler);
        disassembler->disassembly_dex();

        // INFO: xrefs option disabled
        auto analysis = std::make_unique<shuriken::analysis::dex::Analysis>(parsed_dex.get(), disassembler.get(), false);
        auto mm = analysis->get_methods();
        shuriken_opt_log(fmt::format("Printing method names\n"));
        for (auto &[method_name, method_analysis]: mm) {
            shuriken_opt_log(fmt::format("Method name: {}\n", method_name));
            lift_ir(method_analysis, LOGGING);
        }
    }
    return 0;
}

void show_help(std::string &prog_name) {
    fmt::print(fg(fmt::color::green), "USAGE: {} [-h | --help] [-d | --diagnostics] [-f|--file file_name] \n", prog_name);
    fmt::print(fg(fmt::color::green), "    -h | --help: Shows the help menu, like what you're seeing right now\n");
    fmt::print(fg(fmt::color::green), "    -d | --diagnostics: Enables diagnostics for shuriken-opt\n");
    fmt::print(fg(fmt::color::green), "    -f | --file: Analyzes a file with file name\n");
}
/// Simple log for shuriken opt, msgs need to provide newline.
void shuriken_opt_log(const std::string &msg) {
    if (LOGGING)
        fmt::print(stderr, "{}", msg);
}
