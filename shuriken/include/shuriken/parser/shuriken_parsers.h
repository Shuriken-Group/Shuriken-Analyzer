///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author Farenain <kunai.static.analysis@gmail.com>
///
/// @file shuriken_parsers.h
/// @brief Declaration of all the parsers of the library:
/// DEX, APK, Mach-O and IPA

#ifndef SHURIKENLIB_SHURIKEN_PARSERS_H
#define SHURIKENLIB_SHURIKEN_PARSERS_H

#include "shuriken/common/shurikenstream.h"
#include "shuriken/parser/Apk/apk.h"
#include "shuriken/parser/Dex/parser.h"
#include "shuriken/parser/Ipa/ipa.h"
#include <memory>

namespace shuriken::parser {
    std::unique_ptr<dex::Parser> parse_dex(common::ShurikenStream &file);
    std::unique_ptr<dex::Parser> parse_dex(const std::string &file_path);
    dex::Parser *parse_dex(const char *file_path);

    std::unique_ptr<apk::Apk> parse_apk(const std::string &file_path, bool created_xrefs);
    std::unique_ptr<apk::Apk> parse_apk(const char *file_path, bool created_xrefs);

    std::unique_ptr<macho::Parser> parse_macho(common::ShurikenStream &file);
    std::unique_ptr<macho::Parser> parse_macho(const std::string &file_path);
    macho::Parser *parse_macho(const char *file_path);

    std::unique_ptr<ipa::Ipa> parse_ipa(const std::string &file_path);
    std::unique_ptr<ipa::Ipa> parse_ipa(const char *file_path);
}// namespace shuriken::parser

#endif//SHURIKENLIB_SHURIKEN_PARSERS_H
