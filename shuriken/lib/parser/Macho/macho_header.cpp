///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file macho_header.cpp

#include "shuriken/parser/Macho/macho_header.h"

#include "shuriken/common/logger.h"

using namespace shuriken::parser::macho;

#define MH_MAGIC_64 0xfeedfacf
#define CPU_TYPE_ARM64 0x100000C

#define ERROR_MESSAGE(field, expected) "error: '" #field "' is different from '" #expected "'"

void MachoHeader::parse_header(common::ShurikenStream &stream) {

    log(LEVEL::INFO, "Started parsing header");

    // read the macho header
    stream.read_data<machoheader_t>(machoheader_, sizeof(machoheader_t));

    if (machoheader_.magic != MH_MAGIC_64)
        throw std::runtime_error(ERROR_MESSAGE(magic, 0xfeedfacf));

    if (machoheader_.cputype != CPU_TYPE_ARM64)
        throw std::runtime_error(ERROR_MESSAGE(cputype, 0x100000C));

    log(LEVEL::INFO, "Finished parsing header");
}

const MachoHeader::machoheader_t &MachoHeader::get_macho_header_const() const {
    return machoheader_;
}