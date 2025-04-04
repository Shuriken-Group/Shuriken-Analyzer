//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <span>

#include <shuriken/internal/io/shurikenstream.hpp>


namespace shuriken {
namespace dex {

class DexHeader {
private:
    std::uint8_t magic[8];        // Magic bytes from dex, different values are possible
    std::uint32_t checksum;       // Checksum to see if file is correct
    std::uint8_t signature[20];   // Signature of dex
    std::uint32_t file_size;      // Current file size
    std::uint32_t header_size;    // Size of this header
    std::uint32_t endian_tag;     // Type of endianess of the file
    std::uint32_t link_size;      // Data for statically linked files
    std::uint32_t link_off;       // Offset of link data
    std::uint32_t map_off;        // Offset of map data
    std::uint32_t string_ids_size;// Number of DexStrings
    std::uint32_t string_ids_off; // Offset of the DexStrings
    std::uint32_t type_ids_size;  // Number of DexTypes
    std::uint32_t type_ids_off;   // Offset of the DexTypes
    std::uint32_t proto_ids_size; // Number of prototypes
    std::uint32_t proto_ids_off;  // Offset of the prototypes
    std::uint32_t field_ids_size; // Number of fields
    std::uint32_t field_ids_off;  // Offset of the fields
    std::uint32_t method_ids_size;// Number of methods
    std::uint32_t method_ids_off; // Offset of the methods
    std::uint32_t class_defs_size;// Number of class definitions
    std::uint32_t class_defs_off; // Offset of the class definitions
    std::uint32_t data_size;      // Data area size, containing all the support data
    std::uint32_t data_off;       // Offset of data area
public:
    // Constants for array sizes
    static constexpr size_t MAGIC_SIZE = 8;
    static constexpr size_t SIGNATURE_SIZE = 20;

    DexHeader() = default;
    ~DexHeader() = default;

    bool parse(io::ShurikenStream& stream);

    // Getters for each field
    std::span<const std::uint8_t, MAGIC_SIZE> get_magic() const;

    std::uint32_t get_checksum() const;

    std::span<const std::uint8_t, SIGNATURE_SIZE> get_signature() const;

    std::uint32_t get_file_size() const;

    std::uint32_t get_header_size() const;

    std::uint32_t get_endian_tag() const;

    std::uint32_t get_link_size() const;

    std::uint32_t get_link_off() const;

    std::uint32_t get_map_off() const;

    std::uint32_t get_string_ids_size() const;

    std::uint32_t get_string_ids_off() const;

    std::uint32_t get_type_ids_size() const;

    std::uint32_t get_type_ids_off() const;

    std::uint32_t get_proto_ids_size() const;

    std::uint32_t get_proto_ids_off() const;

    std::uint32_t get_field_ids_size() const;

    std::uint32_t get_field_ids_off() const;

    std::uint32_t get_method_ids_size() const;

    std::uint32_t get_method_ids_off() const;

    std::uint32_t get_class_defs_size() const;

    std::uint32_t get_class_defs_off() const;

    std::uint32_t get_data_size() const;

    std::uint32_t get_data_off() const;
};
}
}