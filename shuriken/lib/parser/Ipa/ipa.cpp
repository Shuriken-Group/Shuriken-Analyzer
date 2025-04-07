///--------------------------------------------------------------------*- C++ -*-
/// Shuriken-Analyzer: library for bytecode analysis.
/// @author JorgeRodrigoLeclercq <jorgerodrigoleclercq@gmail.com>
///
/// @file ipa.cpp

#include "shuriken/parser/Ipa/ipa.h"
#include "shuriken/common/extract_file.h"
#include "shuriken/common/logger.h"
#include "shuriken/parser/shuriken_parsers.h"

#include "zip.h"
#include <filesystem>
#include <sstream>

using namespace shuriken::parser::ipa;

// Private API

class Ipa::IpaExtractor {
private:
    /// @brief path to the IPA
    std::string path_to_ipa;
    /// @brief path to the temporal file
    std::string path_to_temporal_file;
    /// @brief zip file
    zip_t *ipa_file;

    /// @brief  names of the Mach-O files in the IPA
    std::vector<std::string_view> macho_file_names;

    /// @brief parsers created during the analysis
    std::unordered_map<std::string,
                       std::unique_ptr<parser::macho::Parser>>
            macho_parsers;
    /// @brief reference for the map, it does not contain
    /// ownership
    std::unordered_map<std::string,
                        std::reference_wrapper<parser::macho::Parser>>
            macho_parsers_s;

public:
    /// @brief public constructor stores the path to the IPA
    IpaExtractor(const char *path_to_ipa) {
        if (!std::filesystem::exists(path_to_ipa))
            throw std::runtime_error("error: IPA provided for the analysis does not exist");
        path_to_temporal_file += std::filesystem::temp_directory_path().string();
        path_to_temporal_file += std::filesystem::path::preferred_separator;
        path_to_temporal_file += std::filesystem::path(path_to_ipa).stem().string();
        std::filesystem::create_directories(path_to_temporal_file);

        // open the IPA
        int error;
        // open the IPA with the zip folder
        ipa_file = zip_open(path_to_ipa, ZIP_RDONLY, &error);
        if (!ipa_file) {
            std::stringstream ss;
            ss << "error: opening the IPA file as a zip: " << error;
            throw std::runtime_error(ss.str());
        }
    }

    /// @brief release zip file and so on...
    ~IpaExtractor() {
        // remove the files
        if (std::filesystem::exists(path_to_temporal_file)) {
            for (const auto &file:
                 std::filesystem::directory_iterator(path_to_temporal_file))
                std::filesystem::remove(file.path().c_str());
        }
        zip_close(ipa_file);
    }

    void analyze_ipa() {
        log(LEVEL::INFO, "Started the IPA analysis of {}", path_to_ipa);

        // get the number of files in the archive
        zip_int64_t num_files = zip_get_num_entries(ipa_file, 0);

        for (zip_int64_t i = 0; i < num_files; ++i) {
            // open the file inside the IPA archive
            zip_file_t* file = zip_fopen_index(ipa_file, i, 0);
            if (!file) continue;

            // read the magic number
            uint32_t magic_number;
            if (zip_fread(file, &magic_number, sizeof(magic_number)) != sizeof(magic_number)) {
                zip_fclose(file);
                continue;
            }
            zip_fclose(file);

            // skip if not a Mach-O file
            if (magic_number != 0xfeedfacf) continue;

            std::string name(zip_get_name(ipa_file, i, 0));
            std::string base_name = std::filesystem::path(name).filename().string();

            // create the file path in the temporal folder
            std::string file_path = path_to_temporal_file;
            file_path += std::filesystem::path::preferred_separator;
            file_path += base_name;

            log(LEVEL::MYDEBUG, "Analyzing a new macho file {}", name);

            // extract the file in the temporal folder
            extract_file(ipa_file, name.c_str(), file_path);

            std::unique_ptr<parser::macho::Parser> current_parser = parse_macho(file_path);

            macho_parsers.insert({name, std::move(current_parser)});
        }
        
        log(LEVEL::INFO, "Finished the analysis of the IPA {}", path_to_ipa);
    }

    std::vector<std::string_view> & get_macho_files_names() {
        if (macho_file_names.empty()) {
            for (auto & parser : macho_parsers)
                macho_file_names.push_back(parser.first);
        }
        return macho_file_names;
    }

    parser::macho::Parser *get_parser_by_file(std::string macho_file) {
        if (!macho_parsers.contains(macho_file)) return nullptr;
        return macho_parsers[macho_file].get();
    }

    std::unordered_map<std::string,
                       std::reference_wrapper<parser::macho::Parser>> &
    get_macho_parsers() {
        if (macho_parsers_s.empty() || macho_parsers_s.size() != macho_parsers.size()) {
            for (auto & parser : macho_parsers) {
                macho_parsers_s.insert({parser.first, std::ref(*parser.second)});
            }
        }
        return macho_parsers_s;
    }

    std::string get_path_to_ipa() const {
        return path_to_ipa;
    }

    std::string get_path_to_temporal_file() const {
        return path_to_temporal_file;
    }
};

// Public API

Ipa::Ipa(std::unique_ptr<IpaExtractor>& ipa_extractor) :
    ipa_extractor_(std::move(ipa_extractor))
{}

Ipa::~Ipa() = default;

std::vector<std::string_view> & Ipa::get_macho_files_names() {
    return ipa_extractor_->get_macho_files_names();
}

shuriken::parser::macho::Parser *Ipa::get_parser_by_file(std::string macho_file) {
    return ipa_extractor_->get_parser_by_file(macho_file);
}

std::unordered_map<std::string,
                   std::reference_wrapper<shuriken::parser::macho::Parser>> &
Ipa::get_macho_parsers() {
    return ipa_extractor_->get_macho_parsers();
}


namespace shuriken {
    namespace parser {
        std::unique_ptr<ipa::Ipa> parse_ipa(const std::string &file_path) {
            std::unique_ptr<Ipa::IpaExtractor> ipa_extractor = std::make_unique<Ipa::IpaExtractor>(file_path.c_str());
            ipa_extractor->analyze_ipa();
            auto ipa = std::make_unique<ipa::Ipa>(ipa_extractor);
            return ipa;
        }

        std::unique_ptr<ipa::Ipa> parse_ipa(const char *file_path) {
            std::unique_ptr<Ipa::IpaExtractor> ipa_extractor = std::make_unique<Ipa::IpaExtractor>(file_path);
            ipa_extractor->analyze_ipa();
            auto ipa = std::make_unique<ipa::Ipa>(ipa_extractor);
            return ipa;
        }
    }// namespace parser
}// namespace shuriken