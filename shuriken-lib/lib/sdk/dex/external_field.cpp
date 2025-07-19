//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/external_field.hpp"
#include "shuriken/internal/providers/dex/dex_external_field_provider.hpp"

using namespace shuriken::dex;

ExternalField::ExternalField(DexExternalFieldProvider &external)
        : dex_external_field_provider(external) {
}

std::string_view ExternalField::get_class_name() const {
    return dex_external_field_provider.get().get_class_name();
}

std::string ExternalField::get_class_name_string() const {
    return dex_external_field_provider.get().get_class_name_string();
}

std::string_view ExternalField::get_name() const {
    return dex_external_field_provider.get().get_name();
}

std::string ExternalField::get_name_string() const {
    return dex_external_field_provider.get().get_name_string();
}

std::string_view ExternalField::get_descriptor() const {
    return dex_external_field_provider.get().get_descriptor();
}

std::string ExternalField::get_descriptor_string() const {
    return dex_external_field_provider.get().get_descriptor_string();
}