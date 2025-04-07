///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq
///
/// @file headers-test-macho.cpp
/// @brief Test the values from the parser and check these
/// values are correct

#include "macho-files-folder.inc"
#include "shuriken/parser/shuriken_parsers.h"

#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstring>

// header data
std::uint32_t magic = 0xfeedfacf;        
std::uint32_t cputype = 0x100000c;      
std::uint32_t cpusubtype = 0x0;   
std::uint32_t filetype = 0x2;     
std::uint32_t ncmds = 59;        
std::uint32_t sizeofcmds = 5312;    
std::uint32_t flags_header = 0x4200085;        
std::uint32_t reserved = 0x0;  

// commands data
uint64_t number_of_loadcommands = 59;
uint64_t number_of_segmentcommands = 5;

uint32_t cmd = 0x19;               
uint32_t cmdsize = 72;           
char segname[16] = "__LINKEDIT";           
uint64_t vmaddr = 0x100020000;             
uint64_t vmsize = 65536;             
uint64_t fileoff = 0x20000;           
uint64_t filesize = 49344;          
uint32_t maxprot = 0x1;           
uint32_t initprot = 0x1;           
uint32_t nsects = 0x0;             
uint32_t flags_seg = 0x0;  

// section data
uint64_t number_of_sections = 23;

char sectname[16] = "__bss";          
char segname_sec[16] = "__DATA";           
uint64_t addr = 0x10001C9C0;              
uint64_t size = 8488;              
uint32_t offset = 0x0;            
uint32_t align = 16;             
uint32_t reloff = 0x0;            
uint32_t nreloc = 0;            
uint32_t flags_sec = 0x1;             
uint32_t reserved1 = 0x0;         
uint32_t reserved2 = 0x0;         
uint32_t reserved3 = 0x0;

void check_macho_header(const shuriken::parser::macho::MachoHeader &header);
void check_macho_loadcommands(const shuriken::parser::macho::MachoCommands &commands);
void check_macho_segmentcommands(const shuriken::parser::macho::MachoCommands &commands);
void check_macho_sections(const shuriken::parser::macho::MachoSections &sections);

int main() {
    std::string test_file = MACHO_FILES_FOLDER
            "ChatGPT";

    std::unique_ptr<shuriken::parser::macho::Parser> macho_parser =
            shuriken::parser::parse_macho(test_file);

    auto &header = macho_parser->get_header();
    check_macho_header(header);

    auto &commands = macho_parser->get_commands();
    check_macho_loadcommands(commands);
    check_macho_segmentcommands(commands);

    auto &sections = macho_parser->get_sections();
    check_macho_sections(sections);

    return 0;
}

void check_macho_header(const shuriken::parser::macho::MachoHeader &header) {
    [[maybe_unused]] auto &macho_header = header.get_macho_header_const();

    assert(magic == macho_header.magic && "error: magic incorrect");
    assert(cputype == macho_header.cputype && "error: cputype incorrect");
    assert(cpusubtype == macho_header.cpusubtype && "error: cpusubtype incorrect");
    assert(filetype == macho_header.filetype && "error: filetype incorrect");
    assert(ncmds == macho_header.ncmds && "error: ncmds incorrect");
    assert(sizeofcmds == macho_header.sizeofcmds && "error: sizeofcmds incorrect");
    assert(flags_header == macho_header.flags && "error flags: incorrect");
    assert(reserved == macho_header.reserved && "error: reserved incorrect");
}

void check_macho_loadcommands(const shuriken::parser::macho::MachoCommands &commands) {
    [[maybe_unused]] auto &macho_loadcommands = commands.get_macho_loadcommands_const();
    
    assert(number_of_loadcommands == macho_loadcommands.size() && "error: number of load commands incorrect");

    [[maybe_unused]] auto &macho_loadcommand = macho_loadcommands.back().get();

    assert(cmd == macho_loadcommand.cmd && "error: cmd incorrect");
    assert(cmdsize == macho_loadcommand.cmdsize && "error: cmdsize incorrect");
}

void check_macho_segmentcommands(const shuriken::parser::macho::MachoCommands &commands) {
    [[maybe_unused]] auto &macho_segmentcommands = commands.get_macho_segmentcommands_const();

    assert(number_of_segmentcommands == macho_segmentcommands.size() && "error: number of segment commands incorrect");

    [[maybe_unused]] auto &macho_segmentcommand = macho_segmentcommands.back().get();

    assert(cmd == macho_segmentcommand.cmd && "error: cmd incorrect");
    assert(cmdsize == macho_segmentcommand.cmdsize && "error: cmdsize incorrect");
    assert(std::strcmp(segname, macho_segmentcommand.segname) == 0 && "error: segname incorrect");
    assert(vmaddr == macho_segmentcommand.vmaddr && "error: vmaddr incorrect");
    assert(vmsize == macho_segmentcommand.vmsize && "error: vmsize incorrect");
    assert(fileoff == macho_segmentcommand.fileoff && "error: fileoff incorrect");
    assert(filesize == macho_segmentcommand.filesize && "error: filesize incorrect");
    assert(maxprot == macho_segmentcommand.maxprot && "error: maxprot incorrect");
    assert(initprot == macho_segmentcommand.initprot && "error: initprot incorrect");
    assert(nsects == macho_segmentcommand.nsects && "error: nsects incorrect");
    assert(flags_seg == macho_segmentcommand.flags && "error: flags incorrect");
}

void check_macho_sections(const shuriken::parser::macho::MachoSections &sections) {
    [[maybe_unused]] auto &macho_sections = sections.get_macho_sections_const();

    assert(number_of_sections == macho_sections.size() && "error: number of sections incorrect");

    [[maybe_unused]] auto &macho_section = macho_sections.back().get();

    assert(std::strcmp(sectname, macho_section.sectname) == 0 && "error: sectname incorrect");
    assert(std::strcmp(segname_sec, macho_section.segname) == 0 && "error: segname incorrect");
    assert(addr == macho_section.addr && "error: addr incorrect");
    assert(size == macho_section.size && "error: size incorrect");
    assert(offset == macho_section.offset && "error: offset incorrect");
    assert(align == macho_section.align && "error: align incorrect");
    assert(reloff == macho_section.reloff && "error: reloff incorrect");
    assert(nreloc == macho_section.nreloc && "error: nreloc incorrect");
    assert(flags_sec == macho_section.flags && "error: flags incorrect");
    assert(reserved1 == macho_section.reserved1 && "error: reserved1 incorrect");
    assert(reserved2 == macho_section.reserved2 && "error: reserved2 incorrect");
    assert(reserved3 == macho_section.reserved3 && "error: reserved3 incorrect");
}