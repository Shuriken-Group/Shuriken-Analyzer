//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/external_field.hpp"
#include "shuriken/internal/sdk/dex/external_field_impl.hpp"

#include <memory>

using namespace shuriken::dex;

ExternalField::ExternalField(ExternalField::Impl * impl)
        : impl(std::unique_ptr<ExternalField::Impl>(impl)) {
}

std::string_view ExternalField::get_class_name() const {
    return impl.get()->get_class_name();
}

std::string ExternalField::get_class_name_string() const {
    return impl.get()->get_class_name_string();
}

std::string_view ExternalField::get_name() const {
    return impl.get()->get_name();
}

std::string ExternalField::get_name_string() const {
    return impl.get()->get_name_string();
}

std::string_view ExternalField::get_descriptor() const {
    return impl.get()->get_descriptor();
}

std::string ExternalField::get_descriptor_string() const {
    return impl.get()->get_descriptor_string();
}