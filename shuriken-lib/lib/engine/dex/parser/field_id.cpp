//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/internal/engine/dex/parser/field_id.hpp>

using namespace shuriken::dex;

FieldID::FieldID(DVMType &class_, DVMType &type_, std::string_view name_) : class_(class_), type_(type_), name_(name_){
}

DVMType &FieldID::get_class() {
    return class_;
}

DVMType &FieldID::get_type() {
    return type_;
}

std::string_view FieldID::get_name() {
    return name_;
}

std::string FieldID::get_name_string() {
    return name_;
}


