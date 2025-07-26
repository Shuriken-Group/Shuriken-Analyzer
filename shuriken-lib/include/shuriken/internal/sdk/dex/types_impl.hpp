//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/constants.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"

#include <memory>

namespace shuriken::dex {
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

    class DVMFundamental::Impl {
    private:
        // @brief Dalvik format of the fundamental type (e.g. L, J, C...)
        const std::string dalvik_format;
        // @brief Canonical format of the fundamental type (e.g. char, short, int, ...)
        std::string canonical_name;
        // @brief Enum representing the fundamental type
        const types::fundamental_e fundamental_type;
    public:
        Impl(std::string_view dalvik_format,
             types::fundamental_e fundamental_type) : dalvik_format(dalvik_format),
                                                      fundamental_type(fundamental_type),
                                                      canonical_name(types::fundamental_s.at(fundamental_type)) {
            canonical_name = types::fundamental_s.at(fundamental_type);
        }

        ~Impl() = default;

        types::type_e get_type() const {
            return types::type_e::FUNDAMENTAL;
        }

        std::string_view get_dalvik_format() const {
            return dalvik_format;
        }

        std::string get_dalvik_format_string() const {
            return dalvik_format;
        }

        std::string_view get_canonical_name() const {
            return canonical_name;
        }

        std::string get_canonical_name_string() const {
            return canonical_name;
        }

        types::fundamental_e get_fundamental_type() const {
            return fundamental_type;
        }
    };

    class DVMClass::Impl {
    private:
        // @brief Dalvik format of the class type (e.g. Ljava/lang/String;)
        const std::string dalvik_format;
        // @brief Canonical format of the class type (e.g. java.lang.String)
        std::string canonical_name;
    public:
        Impl(std::string_view dalvik_format) : dalvik_format(dalvik_format) {
            canonical_name = dalvik_to_canonical(dalvik_format);
        }
        ~Impl() = default;

        types::type_e get_type() const {
            return types::type_e::CLASS;
        }

        std::string_view get_dalvik_format() const {
            return dalvik_format;
        }

        std::string get_dalvik_format_string() const {
            return dalvik_format;
        }

        std::string_view get_canonical_name() const {
            return canonical_name;
        }

        std::string get_canonical_name_string() const {
            return canonical_name;
        }
    };

    class DVMArray::Impl {
    private:
        // @brief Dalvik format of the class type (e.g. Ljava/lang/String;)
        const std::string dalvik_format;
        // @brief Canonical format of the class type (e.g. java.lang.String)
        std::string canonical_name;
        // @brief depth of the array
        const size_t array_depth;
        // @brief base type of the array
        std::unique_ptr<DVMType> base_type;
    public:
        Impl(std::string_view dalvik_format, size_t array_depth,
             DVMType * base_type) : dalvik_format(dalvik_format),
                                                    array_depth(array_depth),
                                                    base_type(base_type) {
            canonical_name = ::shuriken::dex::get_canonical_name_string(*base_type);
            for (size_t i = 0; i < array_depth; ++i) {
                canonical_name += "[]";
            }
        }

        ~Impl() = default;

        std::string_view get_dalvik_format() const {
            return dalvik_format;
        }

        std::string get_dalvik_format_string() const {
            return dalvik_format;
        }

        std::string_view get_canonical_name() const {
            return canonical_name;
        }

        std::string get_canonical_name_string() const {
            return canonical_name;
        }

        size_t get_array_depth() const {
            return array_depth;
        }

        const DVMType &get_base_type() const {
            return *(base_type);
        }
    };
}