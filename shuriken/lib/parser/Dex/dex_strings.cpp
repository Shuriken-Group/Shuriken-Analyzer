//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file strings.cpp

#include "shuriken/parser/Dex/dex_strings.h"
#include "shuriken/common/logger.h"

using namespace shuriken::parser::dex;

void DexStrings::parse_strings(common::ShurikenStream &shuriken_stream,
                               std::uint32_t strings_offset,
                               std::uint32_t n_of_strings) {
    log(LEVEL::INFO, "Start parsing strings");

    auto current_offset = shuriken_stream.tellg();
    std::uint32_t str_offset;// we will read offsets

    // move pointer to the given offset
    shuriken_stream.seekg_safe(strings_offset, std::ios_base::beg);

    // read the DexStrings by offset
    for (size_t I = 0; I < n_of_strings; ++I) {
        shuriken_stream.read_data<std::uint32_t>(str_offset, sizeof(std::uint32_t));

        if (str_offset > shuriken_stream.get_file_size())
            throw std::runtime_error("Error string offset out of bound");

        dex_strings.emplace_back(shuriken_stream.read_dex_string(str_offset));
    }

    shuriken_stream.seekg(current_offset, std::ios_base::beg);
    log(LEVEL::INFO, "Finished parsing strings");
}

void DexStrings::to_xml(std::ofstream &fos) {
    fos << "<DexStrings>\n";
    for (size_t I = 0, E = dex_strings_view.size(); I < E; I++) {
        fos << "\t<string>\n";
        fos << "\t\t<id>" << I << "</id>\n";
        fos << "\t\t<value>" << dex_strings_view[I] << "</value>\n";
        fos << "\t</string>\n";
    }
    fos << "</DexStrings>\n";
}

std::string_view DexStrings::get_string_by_id(std::uint32_t str_id) const {
    if (str_id >= dex_strings.size())
        throw std::runtime_error("Error id of string out of bound");
    return dex_strings.at(str_id);
}

size_t DexStrings::get_number_of_strings() const {
    return dex_strings.size();
}

std::int64_t DexStrings::get_id_by_string(std::string_view str) const {
    auto it = std::ranges::find(dex_strings, str);

    if (it == dex_strings.end())
        return -1;

    return std::distance(dex_strings.begin(), it);
}

const std::vector<std::string_view> DexStrings::get_strings() const {
    if (dex_strings_view.empty() || dex_strings_view.size() != dex_strings.size()) {
        for (auto &str: dex_strings) {
            dex_strings_view.emplace_back(str);
        }
    }
    return dex_strings_view;
}
