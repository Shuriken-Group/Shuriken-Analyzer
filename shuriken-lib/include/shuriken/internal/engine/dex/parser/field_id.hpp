//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include <shuriken/internal/io/shurikenstream.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/dvm_types.hpp>

namespace shuriken {
namespace dex {

class FieldID {
private:
    std::reference_wrapper<DVMType> class_;
    std::reference_wrapper<DVMType> type_;
    std::string name_;
public:
    FieldID(DVMType& class_, DVMType& type_, std::string_view name_);
    ~FieldID() = default;

    DVMType & get_class();

    DVMType & get_type();

    std::string_view get_name();

    std::string get_name_string();
};
}
}