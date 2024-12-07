#pragma once
#include "shuriken/analysis/Dex/dex_analysis.h"


#include <map>

using shuriken::analysis::dex::MethodAnalysis;
void show_help(std::string &prog_name);
bool acquire_input(std::vector<std::string> &args, std::map<std::string, std::string> &options);