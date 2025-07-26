//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/constants.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"

namespace shuriken::dex {

class DVMPrototype::Impl {
private:
    // @brief shorty_idx representing the prototype in a dalvik format
    const std::string shorty_idx;
    // @brief return type of the prototype
    dvmtype_t return_type;
    // @brief all the parameters from the prototype
    std::vector<dvmtype_t> parameter_types;
    // @brief descriptor of the prototype in dalvik format
    std::string descriptor;
public:
    Impl(std::string_view shorty_idx, DVMType &return_type,
         std::vector<dvmtype_t> &parameter_types) :
            shorty_idx(shorty_idx), return_type(return_type),
            parameter_types(std::move(parameter_types)) {
        descriptor = "(";
        for (const auto &type: parameter_types)
            descriptor += get_dalvik_format(type);
        descriptor += ")" + get_dalvik_format(return_type);
    }

    ~Impl() = default;

    std::string_view get_shorty_idx() const {
        return shorty_idx;
    }

    std::string get_shorty_idx_string() const {
        return shorty_idx;
    }

    const DVMType &get_return_type() const {
        return return_type;
    }

    DVMType &get_return_type() {
        return return_type;
    }

    dvmtypes_list_deref_iterator_t get_parameters() {
        static dvmtypes_list_t parameters{parameter_types.data(), parameter_types.size()};
        return parameters;
    }

    std::string_view get_descriptor() {
        return descriptor;
    }

    std::string get_descriptor_string() {
        return descriptor;
    }
};
}