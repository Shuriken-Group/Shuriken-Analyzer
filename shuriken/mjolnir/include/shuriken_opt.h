
#include "shuriken/analysis/Dex/dex_analysis.h"
#include <algorithm>
#include <functional>
#include <iostream>


#include <map>

#include <memory>
#include <shuriken/analysis/Dex/analysis.h>
#include <utility>
#include <variant>
using shuriken::analysis::dex::MethodAnalysis;
void show_help(std::string &prog_name);
bool acquire_input(std::vector<std::string> &args, std::map<std::string, std::string> &options);

using AnalysisClass = shuriken::analysis::dex::Analysis;
using MethodMap = decltype(std::declval<shuriken::analysis::dex::Analysis>().get_methods());
enum OptError {
    GenericError,
};
auto getAnalysis(const std::string &file_name) -> std::variant<std::unique_ptr<AnalysisClass>, OptError>;

auto lift_ir(std::reference_wrapper<MethodAnalysis> mm, const bool LOGGING) -> void;
