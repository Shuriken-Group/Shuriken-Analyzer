///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file header.h
/// @brief MachoHeader of a Mach-O file represented by a structure

#ifndef SHURIKENLIB_MACHO_HEADER_H
#define SHURIKENLIB_MACHO_HEADER_H

#include "shuriken/common/shurikenstream.h"

namespace shuriken::parser::macho {
    class MachoHeader {
    public:
        /// @brief Structure with the definition of the Mach-O header
        /// all these values are later used for parsing the header
        /// from Mach-O
        struct machoheader_t {
            std::uint32_t magic;        //! mach magic number identifier
            std::uint32_t cputype;      //! cpu specifier
            std::uint32_t cpusubtype;   //! machine specifier
            std::uint32_t filetype;     //! type of file
            std::uint32_t ncmds;        //! number of load commands
            std::uint32_t sizeofcmds;   //! the size of all the load commands
            std::uint32_t flags;        //! flags
            std::uint32_t reserved;     //! reserved
        };
    private:
        /// @brief struct with the header from the Mach-O
        struct machoheader_t machoheader_;

    public:
        /// @brief Constructor for the header, default one
        MachoHeader() = default;

        /// @brief Destructor for the header, default one
        ~MachoHeader() = default;

        /// @brief Parse the header from a ShurikenStream file
        /// @param stream ShurikenStream where to read the header.
        void parse_header(common::ShurikenStream &stream);

        /// @brief Obtain a reference of the Mach-O header struct
        /// if not value will be modified, use this function
        /// @return const reference to header structure
        const machoheader_t &get_macho_header_const() const;
    };
}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_HEADER_H