//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include "shuriken/sdk/dex/external_method.hpp"
#include "shuriken/sdk/dex/constants.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"

namespace shuriken::dex {
class ExternalMethod::Impl {
private:
    // @brief name of the method without any type, or class name
    std::string method_name;
    // @brief descriptor name
    std::string descriptor;
    // @brief class name
    std::string class_name;
public:
    Impl(std::string_view method_name,
         std::string_view descriptor,
         std::string_view class_name) :
            method_name(method_name), descriptor(descriptor),
            class_name(class_name) {
    }

    ~Impl() = default;

    std::string_view get_class_name() const {
        return class_name;
    }

    std::string get_class_name_string() const {
        return class_name;
    }

    std::string_view get_name() const {
        return method_name;
    }

    std::string get_name_string() const {
        return method_name;
    }

    std::string_view get_descriptor() const {
        return descriptor;
    }

    std::string get_descriptor_string() const {
        return descriptor;
    }

};
}