///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercqs@gmail.com>
///
/// @file core-api-test.c
/// @brief Test for the Core API in C of the project

#include "ipa-files-folder.inc"
#include "shuriken/parser/shuriken_parsers.h"

#include <iostream>

using namespace shuriken::parser;

const char *file = IPA_FILES_FOLDER
        "ChatGPT.ipa";

int main() {
    std::unique_ptr<ipa::Ipa> parsed_ipa = parse_ipa(file);

    std::vector<std::string_view> macho_files = parsed_ipa->get_macho_files_names();

    std::cout << "Mach-O files:" << std::endl;

    for (const auto& name : macho_files) {
        std::cout << name << std::endl;
    }    
}