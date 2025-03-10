//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file section.h
// @brief MachoSections of a MACHO file represented by a structure

#ifndef SHURIKENLIB_MACHO_SECTIONS_H
#define SHURIKENLIB_MACHO_SECTIONS_H

#include "shuriken/common/shurikenstream.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <memory>
#include <map>

namespace shuriken::parser::macho {
    class MachoSections {
    public:
        /// @brief Structure with the definition of the MACHO section
        /// all these values are later used for parsing the section
        /// from MACHO
        struct section_t {
            char sectname[16];          //! section type
            char segname[16];           //! segment type
            uint64_t addr;              //! starting position of the section
            uint64_t size;              //! size of the section
            uint32_t offset;            //! data's offset of the section
            uint32_t align;             //! alignment in memory, power of 2
            uint32_t reloff;            //! offset to the relocation entries for this section
            uint32_t nreloc;            //! number of relocation entries for this section
            uint32_t flags;             //! flags
            uint32_t reserved1;         //! reserved space for additional info
            uint32_t reserved2;         //! reserved space for additional info
            uint32_t reserved3;         //! reserved space for additional info
        };

        using sections_t = std::vector<std::shared_ptr<section_t>>;
        using segmentsections_t = std::map<uint32_t, sections_t>;

    private:
        /// @brief array with the segment sections
        sections_t sections_;

        /// @brief map with each segment's sections
        segmentsections_t segmentsections_;
    
    public:
        /// @brief Constructor for the sections, default one
        MachoSections() = default;

        /// @brief Destructor for the sections, default one
        ~MachoSections() = default;

        /// @brief Parse the sections from a ShurikenStream file
        /// @param stream ShurikenStream where to read the header
        /// @param number_of_sections number of sections of the segment
        void parse_sections(common::ShurikenStream &stream, 
                            uint32_t number_of_sections, 
                            uint32_t file_offset);

        /// @brief Obtain a reference of the segment sections 
        /// if not value will be modified, use this function
        /// @return const reference to segment sections
        const sections_t &get_sections_const() const;

        /// @brief Obtain a reference of a macho segment's sections
        /// if not value will be modified, use this function
        /// @return const reference to macho segment sections
        const segmentsections_t &get_segmentsections_const() const;
    };
}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_SECTIONS_H