//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file extract_file.h
// @brief Utility to extract zip files such as APKs and IPAs
// types

#ifndef SHURIKENLIB_EXTRACT_FILE_H
#define SHURIKENLIB_EXTRACT_FILE_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

typedef struct zip zip_t;

/// Function to extract a file from a zip archive
bool extract_file(zip_t *archive, const char *filename, const std::string &output_path);

#endif // SHURIKENLIB_EXTRACT_FILE_H