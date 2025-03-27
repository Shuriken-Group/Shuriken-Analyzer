//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/dvm_prototypes.hpp"

using namespace shuriken::dex;

DVMPrototype::DVMPrototype(DVMPrototypeProvider &dvm_prototype_provider) : dvm_prototype_provider(dvm_prototype_provider){
}

std::string_view DVMPrototype::get_shorty_idx() const {
    return std::string_view();
}

std::string DVMPrototype::get_shorty_idx_string() const {
    return std::string();
}

const DVMType *DVMPrototype::get_return_type() const {
    return nullptr;
}

DVMType *DVMPrototype::get_return_type() {
    return nullptr;
}

dvmtypes_list_deref_iterator_t DVMPrototype::get_parameters() {
    dvmtypes_list_t empty {};
    return shuriken::dex::dvmtypes_list_deref_iterator_t(empty);
}


