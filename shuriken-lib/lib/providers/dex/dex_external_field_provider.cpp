//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dex_external_field_provider.hpp"

using namespace shuriken::dex;

DexExternalFieldProvider::DexExternalFieldProvider(std::string_view field_name, std::string_view descriptor,
                                                   std::string_view class_name, DexEngine &dex_engine) :
                                                   dex_engine(dex_engine), field_name(field_name),
                                                   descriptor(descriptor), class_name(class_name) {
}

std::string_view DexExternalFieldProvider::get_class_name() const {
    return class_name;
}

std::string DexExternalFieldProvider::get_class_name_string() const {
    return class_name;
}

std::string_view DexExternalFieldProvider::get_name() const {
    return field_name;
}

std::string DexExternalFieldProvider::get_name_string() const {
    return field_name;
}

std::string_view DexExternalFieldProvider::get_descriptor() const {
    return descriptor;
}

std::string DexExternalFieldProvider::get_descriptor_string() const {
    return descriptor;
}
