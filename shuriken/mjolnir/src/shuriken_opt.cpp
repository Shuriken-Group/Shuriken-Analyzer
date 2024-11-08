#include "shuriken_opt.h"
// Dex & APK stuff
#include "transform/lifter.h"
#include <algorithm>
#include <cstdio>
#include <memory>

#include "fmt/color.h"
#include "fmt/core.h"
#include "shuriken/analysis/Dex/dex_analysis.h"
#include "shuriken/common/logger.h"
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>
#include <string>
#include <variant>
#include <vector>
bool LOGGING = false;
void shuriken_opt_log(const std::string &msg);
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
        auto analysis_opt = getAnalysis(file_name);
        if (std::holds_alternative<OptError>(analysis_opt)) {
            fmt::print(fg(fmt::color::red), "Encountered generic error in getting analysis of {}", file_name);
            return -1;
        }
        auto analysis = std::move(std::get<std::unique_ptr<AnalysisClass>>(analysis_opt));
        auto mm = analysis->get_methods();
        shuriken_opt_log(fmt::format("Printing method names\n"));
        for (auto &[method_name, _]: mm) {
            shuriken_opt_log(fmt::format("Method name: {}\n", method_name));
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
    auto disassembler = std::make_unique<shuriken::disassembler::dex::DexDisassembler>(parsed_dex.get());
    disassembler->disassembly_dex();

    // INFO: xrefs option disabled
    auto analysis = std::make_unique<shuriken::analysis::dex::Analysis>(parsed_dex.get(), disassembler.get(), false);
    if (analysis) return analysis;
    return OptError::GenericError;
}
/// Simple log for shuriken opt, msgs need to provide newline.
void shuriken_opt_log(const std::string &msg) {
    if (LOGGING)
        fmt::print(stderr, "{}", msg);
}
auto getMethods(std::unique_ptr<AnalysisClass> analysis) -> MethodMap {
    return analysis->get_methods();
}
