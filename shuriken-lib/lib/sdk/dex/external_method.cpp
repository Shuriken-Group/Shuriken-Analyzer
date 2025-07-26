//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/external_method.hpp"
#include "shuriken/internal/sdk/dex/external_method_impl.hpp"

#include <memory>

using namespace shuriken::dex;


ExternalMethod::ExternalMethod(ExternalMethod::Impl *impl) : impl(std::unique_ptr<ExternalMethod::Impl>(impl)) {
}

std::string_view ExternalMethod::get_class_name() const {
    return impl.get()->get_class_name();
}

std::string ExternalMethod::get_class_name_string() const {
    return impl.get()->get_class_name_string();
}

std::string_view ExternalMethod::get_name() const {
    return impl.get()->get_name();
}

std::string ExternalMethod::get_name_string() const {
    return impl.get()->get_name_string();
}

std::string_view ExternalMethod::get_descriptor() const {
    return impl.get()->get_descriptor();
}

std::string ExternalMethod::get_descriptor_string() const {
    return impl.get()->get_descriptor_string();
}
