//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file parser.cpp

#include "shuriken/parser/Macho/parser.h"

#include "shuriken/common/logger.h"

using namespace shuriken::parser::macho;

void Parser::parse_macho(common::ShurikenStream &stream) {
    log(LEVEL::INFO, "Start parsing macho file");

    if (stream.get_file_size() < sizeof(MachoHeader::machoheader_t))
        throw std::runtime_error("Error file provided to parser has an incorrect size");

    // parsing of header
    header_.parse_header(stream);

    const auto &macho_header = header_.get_macho_header_const();

    // parsing of the commands
    commands_.parse_commands(stream, 
                            macho_header.ncmds, 
                            sections_);

    log(LEVEL::INFO, "Finished parsing macho file");
}

const MachoHeader &Parser::get_header() const {
    return header_;
}

const MachoCommands &Parser::get_commands() const {
    return commands_;
}

const MachoSections &Parser::get_sections() const {
    return sections_;
}

namespace shuriken {
    namespace parser {
        std::unique_ptr<macho::Parser> parse_macho(common::ShurikenStream &file) {
            auto p = std::make_unique<macho::Parser>();
            p->parse_macho(file);
            return p;
        }

        std::unique_ptr<macho::Parser> parse_macho(const std::string &file_path) {
            std::ifstream ifs(file_path, std::ios::binary);
            common::ShurikenStream file(ifs);

            auto p = std::make_unique<macho::Parser>();
            p->parse_macho(file);
            return p;
        }

        macho::Parser *parse_macho(const char *file_path) {
            std::ifstream ifs(file_path, std::ios::binary);
            common::ShurikenStream file(ifs);

            auto *p = new Parser();
            p->parse_macho(file);
            return p;
        }
    }// namespace parser
}// namespace shuriken