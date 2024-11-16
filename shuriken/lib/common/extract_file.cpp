//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file extract_file.cpp

#include "shuriken/common/extract_file.h"
#include "zip.h"

bool extract_file(zip_t *archive, const char *filename, const std::string &output_path) {
    // Open the file inside the archive
    zip_file_t *file = zip_fopen(archive, filename, 0);
    if (!file) {
        std::cerr << "Failed to open " << filename << " in the archive." << std::endl;
        return false;
    }

    // Get the file info (like size)
    struct zip_stat file_stat;
    zip_stat_init(&file_stat);
    if (zip_stat(archive, filename, 0, &file_stat) == -1) {
        std::cerr << "Failed to get file stats for " << filename << std::endl;
        zip_fclose(file);
        return false;
    }

    // Allocate buffer for file content
    std::vector<char> buffer(file_stat.size);

    // Read the file content
    zip_fread(file, buffer.data(), buffer.size());
    zip_fclose(file);

    // Write the file to the output path
    std::ofstream output_file(output_path, std::ios::binary);
    if (!output_file) {
        std::cerr << "Failed to create output file " << output_path << std::endl;
        return false;
    }
    output_file.write(buffer.data(), buffer.size());
    output_file.close();

    return true;
}