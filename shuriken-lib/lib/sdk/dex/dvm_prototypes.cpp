//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/internal/sdk/dex/prototypes_impl.hpp"

#include <memory>

using namespace shuriken::dex;

DVMPrototype::DVMPrototype(Impl *impl) : impl(std::unique_ptr<Impl>(impl)) {
}

std::string_view DVMPrototype::get_shorty_idx() const {
    return impl.get()->get_shorty_idx();
}

std::string DVMPrototype::get_shorty_idx_string() const {
    return impl.get()->get_shorty_idx_string();
}

const DVMType &DVMPrototype::get_return_type() const {
    return impl.get()->get_return_type();
}

DVMType &DVMPrototype::get_return_type() {
    return impl.get()->get_return_type();
}

dvmtypes_list_deref_iterator_t DVMPrototype::get_parameters() {
    return impl.get()->get_parameters();
}

std::string_view DVMPrototype::get_descriptor() {
    return impl.get()->get_descriptor();
}

std::string DVMPrototype::get_descriptor_string() {
    return impl.get()->get_descriptor_string();
}


