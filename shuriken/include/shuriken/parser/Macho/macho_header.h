//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file header.h
// @brief MachoHeader of a MACHO file represented by a structure

#ifndef SHURIKENLIB_MACHO_HEADER_H
#define SHURIKENLIB_MACHO_HEADER_H

#include "shuriken/common/shurikenstream.h"

#include "cstring"

namespace shuriken::parser::macho {
    class MachoHeader {
    public:
#pragma pack(1)
        /// @brief Structure with the definition of the MACHO header
        /// all these values are later used for parsing the header
        /// from MACHO
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
#pragma pack()
    private:
        /// @brief struct with the header from the macho
        struct machoheader_t machoheader;

    public:
        /// @brief Constructor for the header, default one
        MachoHeader() = default;

        /// @brief Destructor for the header, default one
        ~MachoHeader() = default;

        /// @brief Copy constructor for MachoHeader
        MachoHeader(MachoHeader &header);

        /// @brief Parse the header from a ShurikenStream file
        /// @param stream ShurikenStream where to read the header.
        void parse_header(common::ShurikenStream &stream);
    };
}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_HEADER_H