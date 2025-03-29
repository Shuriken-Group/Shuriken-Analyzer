//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dvm_types_provider.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"

using namespace shuriken::dex;

namespace {
    std::string dalvik_to_canonical(const std::string& dalvik_format) {
        // Skip the leading 'L' and remove trailing ';'
        std::string result = dalvik_format.substr(1, dalvik_format.length() - 2);

        // Replace all '/' with '.'
        size_t pos = 0;
        while ((pos = result.find('/', pos)) != std::string::npos) {
            result.replace(pos, 1, ".");
            pos += 1; // Move past the replacement
        }

        return result;
    }
}

DVMFundamentalProvider::DVMFundamentalProvider(const std::string &dalvik_format,
                                               types::fundamental_e fundamental_type) : dalvik_format(dalvik_format),
                                                                                        fundamental_type(
                                                                                                fundamental_type),
                                                                                                canonical_name(types::fundamental_s.at(fundamental_type)){
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

DVMClassProvider::DVMClassProvider(const std::string &dalvik_format) : dalvik_format(dalvik_format){
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

DVMArrayProvider::DVMArrayProvider(const std::string &dalvik_format, const size_t array_depth,
                                   const DVMType *base_type) : dalvik_format(dalvik_format), array_depth(array_depth), base_type(base_type) {
    if (base_type) {
        canonical_name = ::get_canonical_name_string(*base_type);
        for (int i = 0; i < array_depth; i++) canonical_name += "[]";
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

const DVMType *DVMArrayProvider::get_base_type() const {
    return base_type;
}
