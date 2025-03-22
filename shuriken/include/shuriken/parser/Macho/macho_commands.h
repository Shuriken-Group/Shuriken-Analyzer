//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file header.h
// @brief MachoCommand of a MACHO file represented by a structure

#ifndef SHURIKENLIB_MACHO_COMMANDS_H
#define SHURIKENLIB_MACHO_COMMANDS_H

#include "shuriken/parser/Macho/macho_sections.h"
#include "shuriken/common/shurikenstream.h"

#include <vector>
#include <memory>

namespace shuriken::parser::macho {
    class MachoCommands {
    public:
        /// @brief Structure with the definition of the MACHO load command
        /// all these values are later used for parsing the load command
        /// from MACHO
        struct loadcommand_t {
            uint32_t cmd;               //! command type
            uint32_t cmdsize;           //! command size
        };

        /// @brief Structure with the definition of the MACHO segment load command
        /// all these values are later used for parsing the segment load command
        /// from MACHO
        struct segmentcommand_t {
            uint32_t cmd;               //! command type
            uint32_t cmdsize;           //! command size
            char segname[16];           //! segment type
            uint64_t vmaddr;            //! virtual memory address 
            uint64_t vmsize;            //! size of the segment in the virtual memory
            uint64_t fileoff;           //! offset 
            uint64_t filesize;          //! size of the segment's data
            uint32_t maxprot;           //! maximum VM protection 
            uint32_t initprot;          //! initial VM protection 
            uint32_t nsects;            //! number of sections 
            uint32_t flags;             //! flags 
        };

        using loadcommands_t = std::vector<std::unique_ptr<loadcommand_t>>;
        using loadcommands_s_t = std::vector<std::reference_wrapper<loadcommand_t>>;

        using segmentcommands_t = std::vector<std::unique_ptr<segmentcommand_t>>;
        using segmentcommands_s_t = std::vector<std::reference_wrapper<segmentcommand_t>>;

    private:
        /// @brief array with the load commands from the macho
        loadcommands_t loadcommands;
        mutable loadcommands_s_t loadcommands_s;

        /// @brief array with the segment commands from the macho
        segmentcommands_t segmentcommands;
        mutable segmentcommands_s_t segmentcommands_s;

    public:
        /// @brief Constructor for the load command, default one
        MachoCommands() = default;

        /// @brief Destructor for the load command, default one
        ~MachoCommands() = default;

        /// @brief Parse the load command from a ShurikenStream file
        /// @param stream ShurikenStream where to read the header.
        /// @param number_of_commands number of commands of the Mach-O file
        void parse_commands(common::ShurikenStream &stream, 
                            uint32_t number_of_commands, 
                            MachoSections &sections);

        /// @brief Obtain a reference of the macho load commands 
        /// if not value will be modified, use this function
        /// @return const reference to macho load commands
        const loadcommands_s_t &get_macho_loadcommands_const() const;

        /// @brief Obtain a reference of the macho segment commands
        /// if not value will be modified, use this function
        /// @return const reference to macho segment commands
        const segmentcommands_s_t &get_macho_segmentcommands_const() const;
    };
}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_COMMANDS_H