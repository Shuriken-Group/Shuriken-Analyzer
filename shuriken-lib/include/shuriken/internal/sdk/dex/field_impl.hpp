//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/field.hpp"
#include "shuriken/sdk/dex/constants.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/class.hpp"

namespace shuriken::dex {

class Dex;

class Field::Impl {
private:
    // @brief name of the field without any type, or class name
    std::string field_name;
    // @brief descriptor name like: package/name/class/name->fieldName:type
    std::string descriptor;
    // @brief access flags from the field
    types::access_flags access_flags;
    // @brief access flags as string
    std::string access_flags_str;
    // @brief type of the field from the generation
    types::field_type_e type;
    // @brief DVMType object that represent the type of the field
    std::reference_wrapper<DVMType> field_type;
    // @brief Pointer to owner class (it can be nullptr)
    std::reference_wrapper<Class>  owner_class;
    // @brief Pointer to owner Dex (it can be nullptr)
    std::reference_wrapper<Dex> owner_dex;
public:
    Impl(const std::string &name,
         DVMType &field_type,
         types::access_flags access_flags,
         types::field_type_e type,
         Class &owner_class,
         Dex &owner_dex
         ) : field_name(name), field_type(field_type),
                                  owner_class(owner_class), owner_dex(owner_dex),
                                  access_flags(access_flags),
                                  type(type) {
        descriptor = owner_class.get_name_string() + "->"
                     + name + ":" + get_dalvik_format(field_type);
    }

    ~Impl() = default;

    std::string_view get_name() const {
        return field_name;
    }

    std::string get_name_string() const {
        return field_name;
    }

    types::access_flags get_field_access_flags() const {
        return access_flags;
    }

    std::string_view get_field_access_flags_str() {
        if (access_flags_str.empty()) {
            access_flags_str = access_flags_to_string(access_flags);
        }
        return access_flags_str;
    }

    types::field_type_e get_type() const {
        return type;
    }

    const DVMType &get_field_type() const {
        return field_type;
    }

    DVMType &get_field_type() {
        return field_type;
    }

    const Class &get_owner_class() const {
        return owner_class;
    }

    Class &get_owner_class() {
        return owner_class;
    }

    const Dex &get_owner_dex() const {
        return owner_dex;
    }

    Dex &get_owner_dex() {
        return owner_dex;
    }

    std::string_view get_descriptor() const {
        return descriptor;
    }

    std::string get_descriptor_string() const {
        return descriptor;
    }
};
}