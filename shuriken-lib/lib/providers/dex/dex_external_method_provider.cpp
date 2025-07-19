//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dex_external_method_provider.hpp"

shuriken::dex::DexExternalMethodProvider::DexExternalMethodProvider(std::string_view method_name,
                                                                    std::string_view descriptor,
                                                                    std::string_view class_name,
                                                                    shuriken::dex::DexEngine &dex_engine) :
                                                                    dex_engine(dex_engine), method_name(method_name),
                                                                    descriptor(descriptor), class_name(class_name) {
}


std::string_view shuriken::dex::DexExternalMethodProvider::get_class_name() const {
    return class_name;
}

std::string shuriken::dex::DexExternalMethodProvider::get_class_name_string() const {
    return class_name;
}

std::string_view shuriken::dex::DexExternalMethodProvider::get_name() const {
    return method_name;
}

std::string shuriken::dex::DexExternalMethodProvider::get_name_string() const {
    return method_name;
}

std::string_view shuriken::dex::DexExternalMethodProvider::get_descriptor() const {
    return descriptor;
}

std::string shuriken::dex::DexExternalMethodProvider::get_descriptor_string() const {
    return descriptor;
}
