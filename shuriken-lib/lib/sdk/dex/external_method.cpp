//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/external_method.hpp"
#include "shuriken/internal/providers/dex/dex_external_method_provider.hpp"

using namespace shuriken::dex;


ExternalMethod::ExternalMethod(DexExternalMethodProvider &external) : dex_external_method_provider(external) {
}

std::string_view ExternalMethod::get_class_name() const {
    return dex_external_method_provider.get().get_class_name();
}

std::string ExternalMethod::get_class_name_string() const {
    return dex_external_method_provider.get().get_class_name_string();
}

std::string_view ExternalMethod::get_name() const {
    return dex_external_method_provider.get().get_name();
}

std::string ExternalMethod::get_name_string() const {
    return dex_external_method_provider.get().get_name_string();
}

std::string_view ExternalMethod::get_descriptor() const {
    return dex_external_method_provider.get().get_descriptor();
}

std::string ExternalMethod::get_descriptor_string() const {
    return dex_external_method_provider.get().get_descriptor_string();
}
