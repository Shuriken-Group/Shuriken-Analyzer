///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file section.h
/// @brief MachoSections of a Mach-O file represented by a structure

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
        /// @brief Structure with the definition of the Mach-O section
        /// all these values are later used for parsing the section
        /// from Mach-O
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

        using sections_t = std::vector<std::unique_ptr<section_t>>;
        using sections_s_t = std::vector<std::reference_wrapper<section_t>>;

    private:
        /// @brief array with the segment sections
        sections_t sections_;
        mutable sections_s_t sections_s_;
    
    public:
        /// @brief Constructor for the sections, default one
        MachoSections() = default;

        /// @brief Destructor for the sections, default one
        ~MachoSections() = default;

        /// @brief Parse the sections from a ShurikenStream file
        /// @param stream ShurikenStream where to read the header
        void parse_sections(common::ShurikenStream &stream, 
                            uint32_t number_of_sections);

        /// @brief Obtain a reference of the sections 
        /// if not value will be modified, use this function
        /// @return const reference to sections
        const sections_s_t &get_macho_sections_const() const;
    };
}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_SECTIONS_H