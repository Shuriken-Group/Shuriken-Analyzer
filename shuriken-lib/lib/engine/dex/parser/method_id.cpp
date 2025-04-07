//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/internal/engine/dex/parser/method_id.hpp>

using namespace shuriken::dex;

MethodID::MethodID(DVMType &class_, DVMPrototype &proto_id_, std::string_view name_) : class_(class_),
                                                                                       proto_id_(proto_id_),
                                                                                       name_(name_) {
}

DVMType &MethodID::get_class() {
    return class_;
}

DVMPrototype &MethodID::get_prototype() {
    return proto_id_;
}

std::string_view MethodID::get_name() {
    return name_;
}

std::string MethodID::get_name_string() {
    return name_;
}
