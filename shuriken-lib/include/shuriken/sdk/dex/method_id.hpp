//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include "shuriken/internal/io/shurikenstream.hpp"
#include "custom_types.hpp"
#include "dvm_types.hpp"
#include "dvm_prototypes.hpp"

namespace shuriken {
namespace dex {

class MethodID {
private:
    std::reference_wrapper<DVMType> class_;
    std::reference_wrapper<DVMPrototype> proto_id_;
    std::string name_;
public:
    MethodID(DVMType& class_, DVMPrototype& proto_id_, std::string_view name_);
    ~MethodID() = default;

    DVMType & get_class();

    DVMPrototype & get_prototype();

    std::string_view get_name();

    std::string get_name_string();
};

}
}