#include "shuriken_opt.h"
// Dex & APK stuff
#include <cstdio>

#include "fmt/color.h"
#include <mlir/IR/MLIRContext.h>
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>
#include <string>
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
                fmt::print(fg(fmt::terminal_color::red),
                           "ERROR: Provide input for {}\n", it->first);
                error = true;
            }
        }
    }
    return error;
}
