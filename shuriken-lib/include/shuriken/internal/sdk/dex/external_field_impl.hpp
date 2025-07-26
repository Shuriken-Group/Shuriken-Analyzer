//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/external_field.hpp"
#include "shuriken/sdk/dex/constants.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"

namespace shuriken::dex {
class ExternalField::Impl {
private:
    // @brief name of the field without any type, or class name
    std::string field_name;
    // @brief descriptor name like: package/name/class/name->fieldName:type
    std::string descriptor;
    // @brief class name
    std::string class_name;
public:
    Impl(std::string_view field_name, std::string_view descriptor,
         std::string_view class_name) :
            field_name(field_name),
            descriptor(descriptor), class_name(class_name) {
    }
    ~Impl() = default;

    std::string_view get_class_name() const {
        return class_name;
    }

    std::string get_class_name_string() const {
        return class_name;
    }

    std::string_view get_name() const {
        return field_name;
    }

    std::string get_name_string() const {
        return field_name;
    }

    std::string_view get_descriptor() const {
        return descriptor;
    }

    std::string get_descriptor_string() const {
        return descriptor;
    }

};
}