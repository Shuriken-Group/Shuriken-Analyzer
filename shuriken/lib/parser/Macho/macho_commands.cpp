//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file macho_commands.cpp

#include "shuriken/parser/Macho/macho_commands.h"
#include "shuriken/common/logger.h"

#define LC_SEGMENT 0x19        

using namespace shuriken::parser::macho;

void MachoCommands::parse_commands(common::ShurikenStream &stream, 
                                uint32_t number_of_commands, 
                                MachoSections &sections) {

    log(LEVEL::INFO, "Start parsing commands");
    
    // parse load commands
    for (uint32_t i = 0; i < number_of_commands; ++i) {
        loadcommand_t loadcommand;
        stream.read_data<loadcommand_t>(loadcommand, sizeof(loadcommand_t));
        loadcommands.emplace_back(std::make_unique<loadcommand_t>(loadcommand));

        // parse segment commands
        if (loadcommand.cmd == LC_SEGMENT) {
            stream.seekg(-sizeof(loadcommand), std::ios::cur);
        
            segmentcommand_t segmentcommand;
            stream.read_data<segmentcommand_t>(segmentcommand, sizeof(segmentcommand));
            segmentcommands.emplace_back(std::make_unique<segmentcommand_t>(segmentcommand));
            
            // parse sections
            if (std::string(segmentcommand.segname) == "__DATA" || 
                std::string(segmentcommand.segname) == "__DATA_CONST") { 
                    sections.parse_sections(stream, 
                                            segmentcommand.nsects, 
                                            segmentcommand.fileoff);
            } else {
                stream.seekg(segmentcommand.cmdsize - sizeof(segmentcommand), std::ios::cur);
            }
        } else {
            stream.seekg(loadcommand.cmdsize - sizeof(loadcommand), std::ios::cur);
        }
    }

    log(LEVEL::INFO, "Finished parsing commands");
}

const MachoCommands::loadcommands_s_t &MachoCommands::get_macho_loadcommands_const() const {
    if (loadcommands_s.empty() || loadcommands_s.size() != loadcommands.size()) {
        loadcommands_s.clear();
        for (const auto &entry: loadcommands)
            loadcommands_s.push_back(std::ref(*entry));
    }
    return loadcommands_s;
}

const MachoCommands::segmentcommands_s_t &MachoCommands::get_macho_segmentcommands_const() const {
    if (segmentcommands_s.empty() || segmentcommands_s.size() != segmentcommands.size()) {
        segmentcommands_s.clear();
        for (const auto &entry: segmentcommands)
            segmentcommands_s.push_back(std::ref(*entry));
    }
    return segmentcommands_s;
}