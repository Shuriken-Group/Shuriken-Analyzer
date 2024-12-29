//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file macho_header.cpp

#include "shuriken/parser/Macho/macho_header.h"
#include "shuriken/common/logger.h"

using namespace shuriken::parser::macho;

void MachoHeader::parse_header(common::ShurikenStream &stream) {

    log(LEVEL::INFO, "Start parsing header");

    // read the macho header
    stream.read_data<machoheader_t>(machoheader, sizeof(machoheader_t));

    // print the macho header
	std::cout << "Mach-O Header Information:" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "Magic: 0x" << std::hex << machoheader.magic << std::endl;
    std::cout << "CPU Type: 0x" << std::hex << machoheader.cputype << std::endl;
    std::cout << "CPU Subtype: 0x" << std::hex << machoheader.cpusubtype << std::endl;
    std::cout << "File Type: 0x" << std::hex << machoheader.filetype << std::endl;
    std::cout << "Number of Load Commands: " << std::dec << machoheader.ncmds << std::endl;
    std::cout << "Size of Load Commands: " << machoheader.sizeofcmds << " bytes" << std::endl;
    std::cout << "Flags: 0x" << std::hex << machoheader.flags << std::endl;
    std::cout << "Reserved: 0x" << std::hex << machoheader.reserved << std::endl;

    log(LEVEL::INFO, "Finished parsing header");
}