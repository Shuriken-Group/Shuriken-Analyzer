//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file parser.h
// @brief Parser for the MACHO file, here we contain objects for all the other
// fields

#ifndef SHURIKENLIB_MACHO_PARSER_H
#define SHURIKENLIB_MACHO_PARSER_H

#include "shuriken/common/shurikenstream.h"
#include "shuriken/parser/Macho/macho_header.h"
#include "shuriken/parser/Macho/macho_commands.h"
#include "shuriken/parser/Macho/macho_sections.h"

#include <vector>
#include <memory>

namespace shuriken::parser::macho {

    class Parser {
    private:
        /// @brief MachoHeader of the MACHO file
        MachoHeader header_;

        /// @brief MachoCommands of the MACHO file
        MachoCommands commands_;

        /// @brief MachoSections of the MACHO file
        MachoSections sections_;

    public:
        /// @brief Default constructor of the java
        Parser() = default;
        /// @brief Default destructor of the java
        ~Parser() = default;

        /// @brief parse the macho file from the stream
        /// @param stream stream from where to retrieve the macho data
        void parse_macho(common::ShurikenStream &stream);

        const MachoHeader &get_header() const;

        const MachoCommands &get_commands() const;

        const MachoSections &get_sections() const;
    };

}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_PARSER_H