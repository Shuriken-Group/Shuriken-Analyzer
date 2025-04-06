///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file ipa.h
/// @brief File to analyze IPA files, an IPA file will contain
/// information from all the Mach-O files from the IPA.

#ifndef SHURIKENPROJECT_IPA_H
#define SHURIKENPROJECT_IPA_H

#include "shuriken/parser/Macho/parser.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <map>

namespace shuriken::parser::ipa {

    using namespace shuriken::parser;

    /// @brief an IPA will be the union of different components that
    /// we will work with, altough for the moment we will only focus on Mach-O files
    class Ipa {
    public:
        class IpaExtractor;
    private:
        std::unique_ptr<IpaExtractor> ipa_extractor_;
    public:
        Ipa(std::unique_ptr<IpaExtractor>& ipa_extractor);

        /**
         * @brief destructor for the IPA object, it removes all the
         * temporal files.
         */
        ~Ipa();

        /**
         * @return name of all the Mach-O files found in IPA
         */
        std::vector<std::string_view>& get_macho_files_names();
        
        /**
         * @param macho_file file to retrieve its parser
         * @return pointer to a Parser object, or null
         */
        parser::macho::Parser* get_parser_by_file(std::string macho_file);

        /**
         * @return reference to the map with the parser objects
         */
        std::unordered_map<std::string,
                           std::reference_wrapper<parser::macho::Parser>>&
        get_macho_parsers();
    };
}// namespace shuriken::parser::ipa

#endif//SHURIKENPROJECT_IPA_H