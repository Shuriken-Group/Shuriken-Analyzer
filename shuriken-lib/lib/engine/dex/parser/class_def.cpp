//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/internal/engine/dex/parser/class_def.hpp>
#include <shuriken/sdk/dex/constants.hpp>

using namespace shuriken::dex;

ClassDef::ClassDef() {}

bool ClassDef::parse_class_def(shuriken::io::ShurikenStream &stream, const std::vector<std::string> &string_pools,
                               const std::vector<std::unique_ptr<DVMType>> &types_pool,
                               const std::vector<FieldID> &fields, const std::vector<MethodID> &methods) {
    auto current_offset = stream.position();
    size_t I;
    std::uint32_t size;
    std::uint16_t idx;

    class_idx = stream.read<std::uint32_t>();
    access_flags = stream.read<std::uint32_t>();
    superclass_idx = stream.read<std::uint32_t>();
    interfaces_off = stream.read<std::uint32_t>();
    source_file_idx = stream.read<std::uint32_t>();
    annotations_off = stream.read<std::uint32_t>();
    class_data_off = stream.read<std::uint32_t>();
    static_values_off = stream.read<std::uint32_t>();

    class_type = types_pool[class_idx].get();
    if (superclass_idx != NO_INDEX)
        superclass_type = types_pool[superclass_idx].get();

    if (source_file_idx != NO_INDEX)
        source_file = string_pools[source_file_idx];

    if (interfaces_off) {
        stream.seek(interfaces_off);

        size = stream.read<std::uint32_t>();
        for (I = 0; I < size; I++) {
            idx = stream.read<std::uint16_t>();
            interfaces.emplace_back(types_pool[idx].get());
        }
    }

    stream.seek(current_offset);
    return stream.good();
}


std::uint32_t ClassDef::get_class_idx() const {
    return class_idx;
}

std::uint32_t ClassDef::get_access_flags() const {
    return access_flags;
}

std::uint32_t ClassDef::get_superclass_idx() const {
    return superclass_idx;
}

std::uint32_t ClassDef::get_interfaces_off() const {
    return interfaces_off;
}

std::uint32_t ClassDef::get_source_file_idx() const {
    return source_file_idx;
}

std::uint32_t ClassDef::get_annotations_off() const {
    return annotations_off;
}

std::uint32_t ClassDef::get_class_data_off() const {
    return class_data_off;
}

std::uint32_t ClassDef::get_static_values_off() const {
    return static_values_off;
}

DVMType *ClassDef::get_class_type() const {
    return class_type;
}

DVMType *ClassDef::get_superclass_type() const {
    return superclass_type;
}

std::string_view ClassDef::get_source_file() {
    return source_file;
}

std::string ClassDef::get_source_file_string() {
    return source_file;
}

ClassDef::it_interfaces_list ClassDef::get_interfaces() {
    return make_range(interfaces.begin(), interfaces.end());
}
