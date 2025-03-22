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
    
    std::vector<std::reference_wrapper<section_t>> segmentsections;

    // parse sections
    for (uint32_t i = 0; i < number_of_sections; ++i) {
        section_t section;
        stream.read_data<section_t>(section, sizeof(section_t));
        sections_.emplace_back(std::make_unique<section_t>(section));
        segmentsections.emplace_back(*sections_.back());
    }

    segmentsections_[file_offset] = segmentsections;

    log(LEVEL::INFO, "Finished parsing sections");
}

const MachoSections::sections_s_t &MachoSections::get_sections_const() const {
    if (sections_s_.empty() || sections_s_.size() != sections_.size()) {
        sections_s_.clear();
        for (const auto &entry: sections_)
            sections_s_.push_back(std::ref(*entry));
    }
    return sections_s_;
}

const MachoSections::segmentsections_t &MachoSections::get_segmentsections_const() const {
    return segmentsections_;
}