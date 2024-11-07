#include <algorithm>
#include <iostream>


#include "fmt/color.h"
#include <chrono>
#include <fmt/core.h>
#include <functional>
#include <iostream>
#include <map>
// Dex & APK stuff
#include <shuriken/analysis/Dex/analysis.h>
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>


#include <vector>
void show_help(std::string &prog_name);
bool acquire_input(std::vector<std::string> &args, std::map<std::string, std::string> options);


int main(int argc, char **argv) {
    // list of arguments
    std::vector<std::string> args{argv, argv + argc};

    const std::map<std::string, std::string> options{
            {"-f", ""},
            {"--file", ""},

    };


    // INFO: Check if we need to print out help
    bool need_help = std::find_if(args.begin(), args.end(), [](auto &arg) { return arg == "-h" || arg == "--help"; }) != args.end();

    // INFO: Error is true if something is wrong with input
    bool error = acquire_input(args, options);


    if (need_help || error) {
        show_help(args[0]);
    }


    if (error)
        return -1;

    if (options.at("-f") != "" || options.at("--file") != "") {
        std::string file_name = options.at("-f") != "" ? options.at("-f") : options.at("--file");
    }
    return 0;
}

void show_help(std::string &prog_name) {
    fmt::print(fg(fmt::color::green), "USAGE: {} [-f|--file file_name]\n", prog_name);
    fmt::print(fg(fmt::color::green), "    -h | --help: Shows the help menu, like what you're seeing right now\n");
    fmt::print(fg(fmt::color::green), "    -f | --file: analyze a file with file name\n");
}

bool acquire_input(std::vector<std::string> &args, std::map<std::string, std::string> options) {
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
