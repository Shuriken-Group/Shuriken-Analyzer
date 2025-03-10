//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file macho_sections.cpp

#include "shuriken/parser/Macho/macho_sections.h"
#include "shuriken/common/logger.h"       

using namespace shuriken::parser::macho;

void MachoSections::parse_sections(common::ShurikenStream &stream, 
                                uint32_t number_of_sections, 
                                uint32_t file_offset) {

    log(LEVEL::INFO, "Start parsing sections");
    
    sections_t segmentsections;  

    // parse sections
    for (uint32_t i = 0; i < number_of_sections; ++i) {
        section_t section;
        stream.read_data<section_t>(section, sizeof(section_t));
        sections_.emplace_back(std::make_shared<section_t>(section));
        segmentsections.emplace_back(std::make_shared<section_t>(section));
    }

    segmentsections_[file_offset] = segmentsections;

    log(LEVEL::INFO, "Finished parsing sections");
}

const MachoSections::sections_t &MachoSections::get_sections_const() const {
    return sections_;
}

const MachoSections::segmentsections_t &MachoSections::get_segmentsections_const() const {
    return segmentsections_;
}