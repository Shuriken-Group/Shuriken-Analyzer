
#include <shuriken/internal/engine/dex/parser/dex_header.hpp>
#include <array>

using namespace shuriken::dex;

bool DexHeader::parse(io::ShurikenStream &stream) {
    // Read magic bytes
    for (std::uint8_t &i: magic) {
        i = stream.read<std::uint8_t>();
    }

    // Read checksum
    checksum = stream.read<std::uint32_t>();

    // Read signature
    for (std::uint8_t &i: signature) {
        i = stream.read<std::uint8_t>();
    }

    // Read all remaining header fields
    file_size = stream.read<std::uint32_t>();
    header_size = stream.read<std::uint32_t>();
    endian_tag = stream.read<std::uint32_t>();
    link_size = stream.read<std::uint32_t>();
    link_off = stream.read<std::uint32_t>();
    map_off = stream.read<std::uint32_t>();
    string_ids_size = stream.read<std::uint32_t>();
    string_ids_off = stream.read<std::uint32_t>();
    type_ids_size = stream.read<std::uint32_t>();
    type_ids_off = stream.read<std::uint32_t>();
    proto_ids_size = stream.read<std::uint32_t>();
    proto_ids_off = stream.read<std::uint32_t>();
    field_ids_size = stream.read<std::uint32_t>();
    field_ids_off = stream.read<std::uint32_t>();
    method_ids_size = stream.read<std::uint32_t>();
    method_ids_off = stream.read<std::uint32_t>();
    class_defs_size = stream.read<std::uint32_t>();
    class_defs_off = stream.read<std::uint32_t>();
    data_size = stream.read<std::uint32_t>();
    data_off = stream.read<std::uint32_t>();

    // Check for errors in the stream
    if (!stream.good()) {
        return false;
    }

    // Validate the DEX header
    // Check magic bytes
    static const std::array<std::uint8_t, MAGIC_SIZE> DEX_MAGIC = {'d', 'e', 'x', '\n', '0', '3', '5', '\0'};
    for (auto i = 0; i < MAGIC_SIZE; i++) {
        if (magic[i] != DEX_MAGIC.at(i))
            return false;
    }

    // Check endian tag
    if (endian_tag != 0x12345678 && endian_tag != 0x78563412) {
        // Invalid endian tag
        return false;
    }

    // Basic consistency checks
    if (header_size < sizeof(DexHeader) ||
        file_size == 0 ||
        (string_ids_size > 0 && string_ids_off == 0) ||
        (type_ids_size > 0 && type_ids_off == 0) ||
        (proto_ids_size > 0 && proto_ids_off == 0) ||
        (field_ids_size > 0 && field_ids_off == 0) ||
        (method_ids_size > 0 && method_ids_off == 0) ||
        (class_defs_size > 0 && class_defs_off == 0)) {
        return false;
    }

    return true;
}


std::span<const std::uint8_t, DexHeader::MAGIC_SIZE> DexHeader::get_magic() const {
    return magic;
}

std::uint32_t DexHeader::get_checksum() const {
    return checksum;
}

std::span<const std::uint8_t, DexHeader::SIGNATURE_SIZE> DexHeader::get_signature() const {
    return signature;
}

std::uint32_t DexHeader::get_file_size() const {
    return file_size;
}

std::uint32_t DexHeader::get_header_size() const {
    return header_size;
}

std::uint32_t DexHeader::get_endian_tag() const {
    return endian_tag;
}

std::uint32_t DexHeader::get_link_size() const {
    return link_size;
}

std::uint32_t DexHeader::get_link_off() const {
    return link_off;
}

std::uint32_t DexHeader::get_map_off() const {
    return map_off;
}

std::uint32_t DexHeader::get_string_ids_size() const {
    return string_ids_size;
}

std::uint32_t DexHeader::get_string_ids_off() const {
    return string_ids_off;
}

std::uint32_t DexHeader::get_type_ids_size() const {
    return type_ids_size;
}

std::uint32_t DexHeader::get_type_ids_off() const {
    return type_ids_off;
}

std::uint32_t DexHeader::get_proto_ids_size() const {
    return proto_ids_size;
}

std::uint32_t DexHeader::get_proto_ids_off() const {
    return proto_ids_off;
}

std::uint32_t DexHeader::get_field_ids_size() const {
    return field_ids_size;
}

std::uint32_t DexHeader::get_field_ids_off() const {
    return field_ids_off;
}

std::uint32_t DexHeader::get_method_ids_size() const {
    return method_ids_size;
}

std::uint32_t DexHeader::get_method_ids_off() const {
    return method_ids_off;
}

std::uint32_t DexHeader::get_class_defs_size() const {
    return class_defs_size;
}

std::uint32_t DexHeader::get_class_defs_off() const {
    return class_defs_off;
}

std::uint32_t DexHeader::get_data_size() const {
    return data_size;
}

std::uint32_t DexHeader::get_data_off() const {
    return data_off;
}