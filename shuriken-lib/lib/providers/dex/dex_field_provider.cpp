
#include <shuriken/internal/providers/dex/dex_field_provider.hpp>
#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/sdk/dex/class.hpp>

using namespace shuriken::dex;

namespace {
    std::string access_flags_to_string(uint32_t flags) {
        if (flags == types::access_flags::NONE) {
            return "NONE";
        }

        std::vector<std::string> flag_strings;

        // Check each flag in order of value
        if (flags & types::access_flags::ACC_PUBLIC) flag_strings.push_back("ACC_PUBLIC");
        if (flags & types::access_flags::ACC_PRIVATE) flag_strings.push_back("ACC_PRIVATE");
        if (flags & types::access_flags::ACC_PROTECTED) flag_strings.push_back("ACC_PROTECTED");
        if (flags & types::access_flags::ACC_STATIC) flag_strings.push_back("ACC_STATIC");
        if (flags & types::access_flags::ACC_FINAL) flag_strings.push_back("ACC_FINAL");
        if (flags & types::access_flags::ACC_SYNCHRONIZED) flag_strings.push_back("ACC_SYNCHRONIZED");

        // Handle overlapping values - check context or prioritize
        if (flags & 0x40) {
            // Both ACC_VOLATILE and ACC_BRIDGE have the same value
            // You might want to add logic to distinguish based on context
            flag_strings.push_back("ACC_VOLATILE/ACC_BRIDGE");
        }

        if (flags & 0x80) {
            // Both ACC_TRANSIENT and ACC_VARARGS have the same value
            flag_strings.push_back("ACC_TRANSIENT/ACC_VARARGS");
        }

        if (flags & types::access_flags::ACC_NATIVE) flag_strings.push_back("ACC_NATIVE");
        if (flags & types::access_flags::ACC_INTERFACE) flag_strings.push_back("ACC_INTERFACE");
        if (flags & types::access_flags::ACC_ABSTRACT) flag_strings.push_back("ACC_ABSTRACT");
        if (flags & types::access_flags::ACC_STRICT) flag_strings.push_back("ACC_STRICT");
        if (flags & types::access_flags::ACC_SYNTHETIC) flag_strings.push_back("ACC_SYNTHETIC");
        if (flags & types::access_flags::ACC_ANNOTATION) flag_strings.push_back("ACC_ANNOTATION");
        if (flags & types::access_flags::ACC_ENUM) flag_strings.push_back("ACC_ENUM");
        if (flags & types::access_flags::UNUSED) flag_strings.push_back("UNUSED");
        if (flags & types::access_flags::ACC_CONSTRUCTOR) flag_strings.push_back("ACC_CONSTRUCTOR");
        if (flags & types::access_flags::ACC_DECLARED_SYNCHRONIZED) flag_strings.push_back("ACC_DECLARED_SYNCHRONIZED");

        // Join with pipes
        std::string result;
        for (size_t i = 0; i < flag_strings.size(); ++i) {
            if (i > 0) result += "|";
            result += flag_strings[i];
        }

        return result.empty() ? "NONE" : result;
    }
}

DexFieldProvider::DexFieldProvider(const std::string &name,
                                   DVMType &field_type,
                                   types::access_flags access_flags,
                                   types::field_type_e type,
                                   Class &owner_class,
                                   Dex &owner_dex,
                                   DexEngine &dex_engine) : field_name(name), field_type(field_type),
                                                            owner_class(owner_class), owner_dex(owner_dex),
                                                            access_flags(access_flags), dex_engine(dex_engine),
                                                            type(type) {
    descriptor = owner_class.get_name_string() + "->"
                 + name + ":" + ::get_dalvik_format(field_type);
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

std::string_view DexFieldProvider::get_field_access_flags_str() {
    if (access_flags_str.empty()) {
        access_flags_str = ::access_flags_to_string(access_flags);
    }
    return access_flags_str;
}

types::field_type_e DexFieldProvider::get_type() const {
    return type;
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




