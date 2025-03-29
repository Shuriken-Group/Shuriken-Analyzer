//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/internal/providers/dex/dvm_prototypes_provider.hpp"

using namespace shuriken::dex;

DVMPrototype::DVMPrototype(DVMPrototypeProvider &dvm_prototype_provider) : dvm_prototype_provider(
        dvm_prototype_provider) {
}

std::string_view DVMPrototype::get_shorty_idx() const {
    return dvm_prototype_provider.get().get_shorty_idx();
}

std::string DVMPrototype::get_shorty_idx_string() const {
    return dvm_prototype_provider.get().get_shorty_idx_string();
}

const DVMType &DVMPrototype::get_return_type() const {
    return dvm_prototype_provider.get().get_return_type();
}

DVMType &DVMPrototype::get_return_type() {
    return dvm_prototype_provider.get().get_return_type();
}

dvmtypes_list_deref_iterator_t DVMPrototype::get_parameters() {
    return dvm_prototype_provider.get().get_parameters();
}


