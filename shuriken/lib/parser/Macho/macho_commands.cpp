///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file macho_commands.cpp

#include "shuriken/parser/Macho/macho_commands.h"

#include "shuriken/common/logger.h"

#define LC_SEGMENT 0x19     

#define SEG_TEXT        "__TEXT"
#define SEG_TEXT_EXEC   "__TEXT_EXEC"
#define SEG_DATA        "__DATA"
#define SEG_DATA_CONST  "__DATA_CONST"

using namespace shuriken::parser::macho;

void MachoCommands::parse_commands(common::ShurikenStream &stream, 
                                uint32_t number_of_commands, 
                                MachoSections &sections) {

    log(LEVEL::INFO, "Started parsing commands");
    
    // parsing of the load commands
    for (uint32_t i = 0; i < number_of_commands; ++i) {
        loadcommand_t loadcommand;
        stream.read_data<loadcommand_t>(loadcommand, sizeof(loadcommand_t));
        loadcommands_.emplace_back(std::make_unique<loadcommand_t>(loadcommand));

        // parsing of the segment commands
        if (loadcommand.cmd == LC_SEGMENT) {
            stream.seekg(-sizeof(loadcommand), std::ios::cur);
        
            segmentcommand_t segmentcommand;
            stream.read_data<segmentcommand_t>(segmentcommand, sizeof(segmentcommand));
            segmentcommands_.emplace_back(std::make_unique<segmentcommand_t>(segmentcommand));
            
            // parsing of the sections
            if (std::string(segmentcommand.segname) == SEG_DATA || 
                std::string(segmentcommand.segname) == SEG_DATA_CONST ||
                std::string(segmentcommand.segname) == SEG_TEXT ||
                std::string(segmentcommand.segname) == SEG_TEXT_EXEC) {
                    sections.parse_sections(stream, 
                                            segmentcommand.nsects); 
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
    if (loadcommands_s_.empty() || loadcommands_s_.size() != loadcommands_.size()) {
        loadcommands_s_.clear();
        for (const auto &entry: loadcommands_)
            loadcommands_s_.push_back(std::ref(*entry));
    }
    return loadcommands_s_;
}

const MachoCommands::segmentcommands_s_t &MachoCommands::get_macho_segmentcommands_const() const {
    if (segmentcommands_s_.empty() || segmentcommands_s_.size() != segmentcommands_.size()) {
        segmentcommands_s_.clear();
        for (const auto &entry: segmentcommands_)
            segmentcommands_s_.push_back(std::ref(*entry));
    }
    return segmentcommands_s_;
}