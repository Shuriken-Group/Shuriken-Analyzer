//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain
//
// @file headers-test-macho.cpp
// @brief Test the values from the parser and check these
// values are correct

#include "macho-files-folder.inc"
#include "shuriken/parser/shuriken_parsers.h"

#include <iostream>
#include <cassert>

#include <cstdint>
#include <cstring>

// Useful structures for tests
struct loadcommand_t {
    uint32_t cmd;      
    uint32_t cmdsize;  
};

struct segmentcommand_t {
    uint32_t cmd;               
    uint32_t cmdsize;           
    char segname[16];           
    uint64_t vmaddr;             
    uint64_t vmsize;             
    uint64_t fileoff;           
    uint64_t filesize;          
    uint32_t maxprot;           
    uint32_t initprot;           
    uint32_t nsects;             
    uint32_t flags;             
};

struct section_t {
    char sectname[16];          
    char segname[16];           
    uint64_t addr;              
    uint64_t size;              
    uint32_t offset;            
    uint32_t align;             
    uint32_t reloff;            
    uint32_t nreloc;            
    uint32_t flags;             
    uint32_t reserved1;         
    uint32_t reserved2;         
    uint32_t reserved3;         
};

// header data
std::uint32_t magic = 0xfeedfacf;        
std::uint32_t cputype = 0x100000c;      
std::uint32_t cpusubtype = 0x0;   
std::uint32_t filetype = 0x2;     
std::uint32_t ncmds = 76;        
std::uint32_t sizeofcmds = 8856;    
std::uint32_t flags = 0x218085;        
std::uint32_t reserved = 0x0;     

// load command data
struct loadcommand_t loadcommand = {
    .cmd = 0x1d,
    .cmdsize = 16
};

// segment command data
struct segmentcommand_t segmentcommand = {
    .cmd = 0x19,
    .cmdsize = 72,
    .segname = "__LINKEDIT",
    .vmaddr = 0x10043c000,
    .vmsize = 0x60000,
    .fileoff = 0x438000,
    .filesize = 0x5da50,
    .maxprot = 0x1,
    .initprot = 0x1,
    .nsects = 0x0,
    .flags = 0x0
};

// section data
struct section_t section = {
    .sectname = "__common",
    .segname = "__DATA", 
    .addr = 0x10043a188,
    .size = 569,
    .offset = 0x0,
    .align = 8,
    .reloff = 0x0,
    .nreloc = 0,
    .flags = 0x1,
    .reserved1 = 0x0,
    .reserved2 = 0x0,
    .reserved3 = 0x0
};

void check_header(const shuriken::parser::macho::MachoHeader &header);
void check_loadcommand(const shuriken::parser::macho::MachoCommands &commands);
void check_segmentcommand(const shuriken::parser::macho::MachoCommands &commands);
void check_section(const shuriken::parser::macho::MachoSections &sections);

int main() {
    std::string test_file = MACHO_FILES_FOLDER
            "MachoHeaderParserTest";

    std::unique_ptr<shuriken::parser::macho::Parser> macho_parser =
            shuriken::parser::parse_macho(test_file);

    auto &header = macho_parser->get_header();
    auto &commands = macho_parser->get_commands();
    auto &sections = macho_parser->get_sections();

    check_header(header);
    check_loadcommand(commands);
    check_segmentcommand(commands);
    check_section(sections);

    return 0;
}

void check_header(const shuriken::parser::macho::MachoHeader &header) {
    [[maybe_unused]] auto &macho_header = header.get_macho_header_const();

    assert(magic == macho_header.magic && "Error magic incorrect");
    assert(cputype == macho_header.cputype && "Error cputype incorrect");
    assert(cpusubtype == macho_header.cpusubtype && "Error cpusubtype incorrect");
    assert(filetype == macho_header.filetype && "Error filetype incorrect");
    assert(ncmds == macho_header.ncmds && "Error ncmds incorrect");
    assert(sizeofcmds == macho_header.sizeofcmds && "Error sizeofcmds incorrect");
    assert(flags == macho_header.flags && "Error flags incorrect");
    assert(reserved == macho_header.reserved && "Error reserved incorrect");
}

void check_loadcommand(const shuriken::parser::macho::MachoCommands &commands) {
    [[maybe_unused]] auto &macho_loadcommands = commands.get_macho_loadcommands_const();
    [[maybe_unused]] auto &macho_loadcommand = *macho_loadcommands.back();

    assert(loadcommand.cmd == macho_loadcommand.cmd && "Error: load command cmd incorrect");
    assert(loadcommand.cmdsize == macho_loadcommand.cmdsize && "Error: load command cmdsize incorrect");
}

void check_segmentcommand(const shuriken::parser::macho::MachoCommands &commands) {
    [[maybe_unused]] auto &macho_segmentcommands = commands.get_macho_segmentcommands_const();
    [[maybe_unused]] auto &macho_segmentcommand = *macho_segmentcommands.back();

    assert(segmentcommand.cmd == macho_segmentcommand.cmd && "Error: segment command cmd incorrect");
    assert(segmentcommand.cmdsize == macho_segmentcommand.cmdsize && "Error: segment command cmdsize incorrect");
    assert(std::strcmp(segmentcommand.segname, macho_segmentcommand.segname) == 0 && "Error: segment command segname incorrect");
    assert(segmentcommand.vmaddr == macho_segmentcommand.vmaddr && "Error: segment command vmaddr incorrect");
    assert(segmentcommand.vmsize == macho_segmentcommand.vmsize && "Error: segment command vmsize incorrect");
    assert(segmentcommand.fileoff == macho_segmentcommand.fileoff && "Error: segment command fileoff incorrect");
    assert(segmentcommand.filesize == macho_segmentcommand.filesize && "Error: segment command filesize incorrect");
    assert(segmentcommand.maxprot == macho_segmentcommand.maxprot && "Error: segment command maxprot incorrect");
    assert(segmentcommand.initprot == macho_segmentcommand.initprot && "Error: segment command initprot incorrect");
    assert(segmentcommand.nsects == macho_segmentcommand.nsects && "Error: segment command nsects incorrect");
    assert(segmentcommand.flags == macho_segmentcommand.flags && "Error: segment command flags incorrect");
}

void check_section(const shuriken::parser::macho::MachoSections &sections) {
    [[maybe_unused]] auto &macho_sections = sections.get_sections_const();
    [[maybe_unused]] auto &macho_section = *macho_sections.back();

    assert(std::strcmp(section.sectname, macho_section.sectname) == 0 && "Error: section sectname incorrect");
    assert(std::strcmp(section.segname, macho_section.segname) == 0 && "Error: section segname incorrect");
    assert(section.addr == macho_section.addr && "Error: section addr incorrect");
    assert(section.size == macho_section.size && "Error: section size incorrect");
    assert(section.offset == macho_section.offset && "Error: section offset incorrect");
    assert(section.align == macho_section.align && "Error: section align incorrect");
    assert(section.reloff == macho_section.reloff && "Error: section reloff incorrect");
    assert(section.nreloc == macho_section.nreloc && "Error: section nreloc incorrect");
    assert(section.flags == macho_section.flags && "Error: section flags incorrect");
    assert(section.reserved1 == macho_section.reserved1 && "Error: section reserved1 incorrect");
    assert(section.reserved2 == macho_section.reserved2 && "Error: section reserved2 incorrect");
    assert(section.reserved3 == macho_section.reserved3 && "Error: section reserved3 incorrect");
}