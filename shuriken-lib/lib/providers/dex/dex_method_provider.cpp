//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dex_method_provider.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/class.hpp"

using namespace shuriken::dex;

shuriken::dex::DexMethodProvider::DexMethodProvider(const std::string &name,
                                                    types::access_flags access_flags,
                                                    DVMPrototype &method_prototype,
                                                    Class &owner_class,
                                                    Dex &owner_dex,
                                                    DexEngine &dex_engine)
        : method_name(name),
          method_access_flags(access_flags), method_prototype(method_prototype),
          owner_class(owner_class), owner_dex(owner_dex),
          dex_engine(dex_engine) {
    method_descriptor = owner_class.get_name_string() + "->"
                        + name + method_prototype.get_descriptor_string();
}


std::string_view shuriken::dex::DexMethodProvider::get_name() const {
    return method_name;
}


std::string shuriken::dex::DexMethodProvider::get_name_string() const {
    return method_name;
}


shuriken::dex::types::access_flags shuriken::dex::DexMethodProvider::get_method_access_flags() const {
    return method_access_flags;
}


const shuriken::dex::DVMPrototype &shuriken::dex::DexMethodProvider::get_method_prototype() const {
    return method_prototype;
}


shuriken::dex::DVMPrototype &shuriken::dex::DexMethodProvider::get_method_prototype() {
    return method_prototype;
}


const shuriken::dex::Class &shuriken::dex::DexMethodProvider::get_owner_class() const {
    return owner_class;
}


shuriken::dex::Class &shuriken::dex::DexMethodProvider::get_owner_class() {
    return owner_class;
}

const shuriken::dex::Dex &shuriken::dex::DexMethodProvider::get_owner_dex() const {
    return owner_dex;
}


shuriken::dex::Dex &shuriken::dex::DexMethodProvider::get_owner_dex() {
    return owner_dex;
}


std::string_view shuriken::dex::DexMethodProvider::get_descriptor() const {
    return method_descriptor;
}


std::string shuriken::dex::DexMethodProvider::get_descriptor_string() const {
    return method_descriptor;
}


std::uint16_t shuriken::dex::DexMethodProvider::registers_size() const {
    return number_of_registers;
}


std::span<const std::uint8_t> shuriken::dex::DexMethodProvider::get_bytecode() const {
    static std::span<const std::uint8_t> data {bytecode};
    return data;
}