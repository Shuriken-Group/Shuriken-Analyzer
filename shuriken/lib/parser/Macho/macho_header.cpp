//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file macho_header.cpp

#include "shuriken/parser/Macho/macho_header.h"
#include "shuriken/common/logger.h"

using namespace shuriken::parser::macho;

#define MH_MAGIC_64 0xfeedfacf
#define CPU_TYPE_ARM64 0x100000C
#define MH_EXECUTE 0x2

#define ERROR_MESSAGE(field, expected) "Error '" #field "' is different from '" #expected "'"

void MachoHeader::parse_header(common::ShurikenStream &stream) {

    log(LEVEL::INFO, "Start parsing header");

    // read the macho header
    stream.read_data<machoheader_t>(machoheader, sizeof(machoheader_t));

    if (machoheader.magic != MH_MAGIC_64)
        throw std::runtime_error(ERROR_MESSAGE(magic, 0xfeedfacf));

    if (machoheader.cputype != CPU_TYPE_ARM64)
        throw std::runtime_error(ERROR_MESSAGE(cputype, 0x100000C));

    if (machoheader.filetype != MH_EXECUTE)
        throw std::runtime_error(ERROR_MESSAGE(filetype, 0x2));

    log(LEVEL::INFO, "Finished parsing header");
}

const MachoHeader::machoheader_t &MachoHeader::get_macho_header_const() const {
    return machoheader;
}