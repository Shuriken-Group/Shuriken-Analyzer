///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file header.h
/// @brief MachoCommand of a Mach-O file represented by a structure

#ifndef SHURIKENLIB_MACHO_COMMANDS_H
#define SHURIKENLIB_MACHO_COMMANDS_H

#include "shuriken/parser/Macho/macho_sections.h"
#include "shuriken/common/shurikenstream.h"

#include <vector>
#include <memory>

namespace shuriken::parser::macho {
    class MachoCommands {
    public:
        /// @brief Structure with the definition of the Mach-O load command
        /// all these values are later used for parsing the load command
        /// from Mach-O
        struct loadcommand_t {
            uint32_t cmd;               //! command type
            uint32_t cmdsize;           //! command size
        };

        /// @brief Structure with the definition of the Mach-O segment load command
        /// all these values are later used for parsing the segment load command
        /// from Mach-O
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
        /// @brief array with the load commands from the Mach-O
        loadcommands_t loadcommands_;
        mutable loadcommands_s_t loadcommands_s_;

        /// @brief array with the segment commands from the Mach-O
        segmentcommands_t segmentcommands_;
        mutable segmentcommands_s_t segmentcommands_s_;

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

        /// @brief Obtain a reference of the Mach-O load commands 
        /// if not value will be modified, use this function
        /// @return const reference to Mach-O load commands
        const loadcommands_s_t &get_macho_loadcommands_const() const;

        /// @brief Obtain a reference of the Mach-O segment commands
        /// if not value will be modified, use this function
        /// @return const reference to Mach-O segment commands
        const segmentcommands_s_t &get_macho_segmentcommands_const() const;
    };
}// namespace shuriken::parser::macho

#endif//SHURIKENLIB_MACHO_COMMANDS_H