//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dvm_prototypes_provider.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"

using namespace shuriken::dex;

DVMPrototypeProvider::DVMPrototypeProvider(const std::string &shorty_idx, DVMType &return_type,
                                           std::vector<dvmtype_t> &parameter_types) :
        shorty_idx(shorty_idx), return_type(return_type),
        parameter_types(std::move(parameter_types)) {
}

std::string_view DVMPrototypeProvider::get_shorty_idx() const {
    return shorty_idx;
}

std::string DVMPrototypeProvider::get_shorty_idx_string() const {
    return shorty_idx;
}

const DVMType &DVMPrototypeProvider::get_return_type() const {
    return return_type;
}

DVMType &DVMPrototypeProvider::get_return_type() {
    return return_type;
}

dvmtypes_list_deref_iterator_t DVMPrototypeProvider::get_parameters() {
    dvmtypes_list_t parameters{parameter_types.data(), parameter_types.size()};
    return parameters;
}

