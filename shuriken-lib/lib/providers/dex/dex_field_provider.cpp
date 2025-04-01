
#include <shuriken/internal/providers/dex/dex_field_provider.hpp>
#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/sdk/dex/class.hpp>

using namespace shuriken::dex;

DexFieldProvider::DexFieldProvider(const std::string &name,
                                   DVMType &type,
                                   types::access_flags access_flags,
                                   Class &owner_class,
                                   Dex &owner_dex,
                                   DexEngine &dex_engine) : field_name(name), field_type(type),
                                                            owner_class(owner_class), owner_dex(owner_dex),
                                                            access_flags(access_flags), dex_engine(dex_engine) {
    descriptor = owner_class.get_name_string() + "->"
                 + name + ":" + ::get_dalvik_format(type);
}

std::string_view DexFieldProvider::get_name() const {
    return field_name;
}

std::string DexFieldProvider::get_name_string() const {
    return field_name;
}

types::access_flags DexFieldProvider::get_field_access_flags() const {
    return access_flags;
}

const DVMType &DexFieldProvider::get_field_type() const {
    return field_type;
}

DVMType &DexFieldProvider::get_field_type() {
    return field_type;
}

const Class &DexFieldProvider::get_owner_class() const {
    return owner_class;
}

Class &DexFieldProvider::get_owner_class() {
    return owner_class;
}

const Dex &DexFieldProvider::get_owner_dex() const {
    return owner_dex;
}

Dex &DexFieldProvider::get_owner_dex() {
    return owner_dex;
}

std::string_view DexFieldProvider::get_descriptor() const {
    return descriptor;
}

std::string DexFieldProvider::get_descriptor_string() const {
    return descriptor;
}




