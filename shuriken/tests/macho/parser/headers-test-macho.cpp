//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file headers-test-macho.cpp
// @brief Test the values from the parser and check these
// values are correct

#include "macho-files-folder.inc"
#include "shuriken/common/shurikenstream.h"
#include "shuriken/parser/Macho/macho_header.h"

#include <iostream>
#include <cassert>


// header data
std::uint32_t magic = 0xfeedfacf;        
std::uint32_t cputype = 0x100000c;      
std::uint32_t cpusubtype = 0x0;   
std::uint32_t filetype = 0x2;     
std::uint32_t ncmds = 76;        
std::uint32_t sizeofcmds = 8856;    
std::uint32_t flags = 0x218085;        
std::uint32_t reserved = 0x0;     


void check_header(const shuriken::parser::macho::MachoHeader::machoheader_t &header);

int main() {
    std::string test_path = MACHO_FILES_FOLDER
            "MachoHeaderParserTest";
    std::ifstream test_file(test_path);
    shuriken::common::ShurikenStream test_stream(test_file);

    shuriken::parser::macho::MachoHeader macho_header;
    macho_header.parse_header(test_stream);
    auto &header = macho_header.get_macho_header_const();

    check_header(header);

    return 0;
}


void check_header(const shuriken::parser::macho::MachoHeader::machoheader_t &header) {
    assert(magic == header.magic && "Error magic incorrect");
    assert(cputype == header.cputype && "Error cputype incorrect");
    assert(cpusubtype == header.cpusubtype && "Error cpusubtype incorrect");
    assert(filetype == header.filetype && "Error filetype incorrect");
    assert(ncmds == header.ncmds && "Error ncmds incorrect");
    assert(sizeofcmds == header.sizeofcmds && "Error sizeofcmds incorrect");
    assert(flags == header.flags && "Error flags incorrect");
    assert(reserved == header.reserved && "Error reserved incorrect");
}