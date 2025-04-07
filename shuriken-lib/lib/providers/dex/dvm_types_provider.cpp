//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dvm_types_provider.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"

using namespace shuriken::dex;

namespace {
    std::string dalvik_to_canonical(std::string_view dalvik_format) {
        // Skip the leading 'L' and remove trailing ';'
        std::string result = std::string(dalvik_format.substr(1, dalvik_format.length() - 2));

        // Replace all '/' with '.'
        size_t pos = 0;
        while ((pos = result.find('/', pos)) != std::string::npos) {
            result.replace(pos, 1, ".");
            pos += 1; // Move past the replacement
        }

        return result;
    }
}

DVMFundamentalProvider::DVMFundamentalProvider(std::string_view dalvik_format,
                                               types::fundamental_e fundamental_type) : dalvik_format(dalvik_format),
                                                                                        fundamental_type(
                                                                                                fundamental_type),
                                                                                        canonical_name(
                                                                                                types::fundamental_s.at(
                                                                                                        fundamental_type)) {
    canonical_name = types::fundamental_s.at(fundamental_type);
}

types::type_e DVMFundamentalProvider::get_type() const {
    return types::type_e::FUNDAMENTAL;
}

std::string_view DVMFundamentalProvider::get_dalvik_format() const {
    return dalvik_format;
}

std::string DVMFundamentalProvider::get_dalvik_format_string() const {
    return dalvik_format;
}

std::string_view DVMFundamentalProvider::get_canonical_name() const {
    return canonical_name;
}

std::string DVMFundamentalProvider::get_canonical_name_string() const {
    return canonical_name;
}

types::fundamental_e DVMFundamentalProvider::get_fundamental_type() const {
    return fundamental_type;
}

DVMClassProvider::DVMClassProvider(std::string_view dalvik_format) : dalvik_format(dalvik_format) {
    canonical_name = ::dalvik_to_canonical(dalvik_format);
}

types::type_e DVMClassProvider::get_type() const {
    return types::type_e::CLASS;
}

std::string_view DVMClassProvider::get_dalvik_format() const {
    return dalvik_format;
}

std::string DVMClassProvider::get_dalvik_format_string() const {
    return dalvik_format;
}

std::string_view DVMClassProvider::get_canonical_name() const {
    return canonical_name;
}

std::string DVMClassProvider::get_canonical_name_string() const {
    return canonical_name;
}

DVMArrayProvider::DVMArrayProvider(std::string_view dalvik_format, size_t array_depth,
                                   DVMTypeProvider* base_type_provider) : dalvik_format(dalvik_format),
                                                                                          array_depth(array_depth),
                                                                                          base_type_provider(
                                                                                                  base_type_provider) {
    canonical_name = ::get_canonical_name_string(*base_type_provider);
    for (int i = 0; i < array_depth; i++) canonical_name += "[]";

    if (::get_type(*base_type_provider) == types::type_e::FUNDAMENTAL) {
        DVMFundamentalProvider* fundamental = ::as_fundamental(*base_type_provider);
        base_type = std::make_unique<DVMType>(DVMFundamental(*fundamental));
    } else if (::get_type(*base_type_provider) == types::type_e::CLASS) {
        DVMClassProvider* class_provider = ::as_class(*base_type_provider);
        base_type = std::make_unique<DVMType>(DVMClass(*class_provider));
    } else {
        DVMArrayProvider* array_provider = ::as_array(*base_type_provider);
        base_type = std::make_unique<DVMType>(DVMArray(*array_provider));
    }
}

types::type_e DVMArrayProvider::get_type() const {
    return types::type_e::ARRAY;
}

std::string_view DVMArrayProvider::get_dalvik_format() const {
    return dalvik_format;
}

std::string DVMArrayProvider::get_dalvik_format_string() const {
    return dalvik_format;
}

std::string_view DVMArrayProvider::get_canonical_name() const {
    return canonical_name;
}

std::string DVMArrayProvider::get_canonical_name_string() const {
    return canonical_name;
}

size_t DVMArrayProvider::get_array_depth() const {
    return array_depth;
}

const DVMType &DVMArrayProvider::get_base_type() const {
    return *(base_type);
}


// Helper template for std::visit
template<class... Ts>
struct overloaded : Ts ... {
    using Ts::operator()...;
};

// Deduction guide (C++17 or later)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;


types::type_e shuriken::dex::get_type(const DVMTypeProvider &type) {
    return std::visit(overloaded{
            [](const DVMFundamentalProvider &) { return types::type_e::FUNDAMENTAL; },
            [](const DVMClassProvider &) { return types::type_e::CLASS; },
            [](const DVMArrayProvider &) { return types::type_e::ARRAY; },
    }, type);
}

std::string_view shuriken::dex::get_dalvik_format_string(const DVMTypeProvider &type) {
    return std::visit(overloaded{
            [](const DVMFundamentalProvider &t) { return t.get_dalvik_format(); },
            [](const DVMClassProvider &t) { return t.get_dalvik_format(); },
            [](const DVMArrayProvider &t) { return t.get_dalvik_format(); },
    }, type);
}

std::string shuriken::dex::get_dalvik_format(const DVMTypeProvider &type) {
    return std::visit(overloaded{
            [](const DVMFundamentalProvider &t) { return t.get_dalvik_format_string(); },
            [](const DVMClassProvider &t) { return t.get_dalvik_format_string(); },
            [](const DVMArrayProvider &t) { return t.get_dalvik_format_string(); },
    }, type);
}

std::string_view shuriken::dex::get_canonical_name(const DVMTypeProvider &type) {
    return std::visit(overloaded{
            [](const DVMFundamentalProvider &t) { return t.get_canonical_name(); },
            [](const DVMClassProvider &t) { return t.get_canonical_name(); },
            [](const DVMArrayProvider &t) { return t.get_canonical_name(); },
    }, type);
}

std::string shuriken::dex::get_canonical_name_string(const DVMTypeProvider &type) {
    return std::visit(overloaded{
            [](const DVMFundamentalProvider &t) { return t.get_canonical_name_string(); },
            [](const DVMClassProvider &t) { return t.get_canonical_name_string(); },
            [](const DVMArrayProvider &t) { return t.get_canonical_name_string(); },
    }, type);
}

DVMFundamentalProvider *shuriken::dex::as_fundamental(DVMTypeProvider &type) {
    return std::get_if<DVMFundamentalProvider>(&type);
}

DVMClassProvider *shuriken::dex::as_class(DVMTypeProvider &type) {
    return std::get_if<DVMClassProvider>(&type);
}

DVMArrayProvider *shuriken::dex::as_array(DVMTypeProvider &type) {
    return std::get_if<DVMArrayProvider>(&type);
}