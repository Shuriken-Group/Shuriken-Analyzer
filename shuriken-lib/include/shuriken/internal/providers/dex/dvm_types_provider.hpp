//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <memory>
#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/internal/providers/dex/custom_types.hpp>

namespace shuriken {
namespace dex {

class DVMFundamentalProvider {
private:
    // @brief Dalvik format of the fundamental type (e.g. L, J, C...)
    const std::string dalvik_format;
    // @brief Canonical format of the fundamental type (e.g. char, short, int, ...)
    std::string canonical_name;
    // @brief Enum representing the fundamental type
    const types::fundamental_e fundamental_type;

public:
    DVMFundamentalProvider(std::string_view dalvik_format, types::fundamental_e fundamental_type);
    ~DVMFundamentalProvider() = default;

    /**
     * @brief Gets the general type enum value of this fundamental type
     * @return The type_e enum value representing this type
     */
    types::type_e get_type() const;

    /**
     * @brief Gets the Dalvik format character representation as a string view
     * @return A string view to the internal Dalvik format representation (e.g., "I", "J", "Z")
     * @note Does not allocate memory as it returns a view to internal storage
     */
    std::string_view get_dalvik_format() const;

    /**
     * @brief Gets a copy of the Dalvik format character representation
     * @return A string containing the Dalvik format representation
     * @note Allocates memory for the returned string
     */
    std::string get_dalvik_format_string() const;

    /**
     * @brief Gets the canonical Java type name as a string view
     * @return A string view to the internal canonical type name (e.g., "int", "long", "boolean")
     * @note Does not allocate memory as it returns a view to internal storage
     */
    std::string_view get_canonical_name() const;

    /**
     * @brief Gets a copy of the canonical Java type name
     * @return A string containing the canonical type name
     * @note Allocates memory for the returned string
     */
    std::string get_canonical_name_string() const;

    /**
     * @brief Gets the specific fundamental type enum value
     * @return The fundamental_e enum value representing this specific fundamental type
     */
    types::fundamental_e get_fundamental_type() const;

};

class DVMClassProvider {
private:
    // @brief Dalvik format of the class type (e.g. Ljava/lang/String;)
    const std::string dalvik_format;
    // @brief Canonical format of the class type (e.g. java.lang.String)
    std::string canonical_name;
public:
    /**
     * @brief Constructs a class provider from a Dalvik format string
     * @param dalvik_format The Dalvik format of the class (e.g. "Ljava/lang/String;")
     */
    DVMClassProvider(std::string_view dalvik_format);
    ~DVMClassProvider() = default;

    /**
     * @brief Gets the general type enum value
     * @return Always returns types::type_e::CLASS
     */
    types::type_e get_type() const;

    /**
     * @brief Gets the Dalvik format as a string view
     * @return A string view to the internal Dalvik format representation
     * @note Does not allocate memory as it returns a view to internal storage
     */
    std::string_view get_dalvik_format() const;

    /**
     * @brief Gets a copy of the Dalvik format
     * @return A string containing the Dalvik format
     * @note Allocates memory for the returned string
     */
    std::string get_dalvik_format_string() const;

    /**
     * @brief Gets the canonical class name as a string view
     * @return A string view to the internal canonical name (e.g., "java.lang.String")
     * @note Does not allocate memory as it returns a view to internal storage
     */
    std::string_view get_canonical_name() const;

    /**
     * @brief Gets a copy of the canonical class name
     * @return A string containing the canonical name
     * @note Allocates memory for the returned string
     */
    std::string get_canonical_name_string() const;
};

class DVMArrayProvider {
private:
    // @brief Dalvik format of the class type (e.g. Ljava/lang/String;)
    const std::string dalvik_format;
    // @brief Canonical format of the class type (e.g. java.lang.String)
    std::string canonical_name;
    // @brief depth of the array
    const size_t array_depth;
    // @brief keep a unique_ptr of the type
    std::unique_ptr<DVMTypeProvider> base_type_provider;
    // @brief base type of the array
    std::unique_ptr<DVMType> base_type;
public:
    /**
     * @brief Constructs an array provider from its components
     * @param dalvik_format The Dalvik format of the array type
     * @param array_depth The nesting depth of the array
     * @param base_type Pointer to the base type of the array elements
     */
    DVMArrayProvider(std::string_view dalvik_format, size_t array_depth, DVMTypeProvider* base_type_provider);
    ~DVMArrayProvider() = default;

    /**
     * @brief Gets the general type enum value
     * @return Always returns types::type_e::ARRAY
     */
    types::type_e get_type() const;

    /**
     * @brief Gets the Dalvik format as a string view
     * @return A string view to the internal Dalvik format representation
     * @note Does not allocate memory as it returns a view to internal storage
     */
    std::string_view get_dalvik_format() const;

    /**
     * @brief Gets a copy of the Dalvik format
     * @return A string containing the Dalvik format
     * @note Allocates memory for the returned string
     */
    std::string get_dalvik_format_string() const;

    /**
     * @brief Gets the canonical array type name as a string view
     * @return A string view to the internal canonical name
     * @note Does not allocate memory as it returns a view to internal storage
     */
    std::string_view get_canonical_name() const;

    /**
     * @brief Gets a copy of the canonical array type name
     * @return A string containing the canonical name
     * @note Allocates memory for the returned string
     */
    std::string get_canonical_name_string() const;

    /**
     * @brief Gets the nesting depth of the array
     * @return The depth of the array (number of dimensions)
     */
    size_t get_array_depth() const;

    /**
     * @brief Gets the base type of the array elements
     * @return Pointer to the DVMType representing the array's element type
     */
    const DVMType& get_base_type() const;
};

/**
 * @param type type to obtain its main type_e
 * @return type_e of provided object
 */
types::type_e get_type(const DVMTypeProvider & type);

/**
 * @param type type to obtain its name in dalvik format
 * @return dalvik format of type as string_view
 */
std::string_view get_dalvik_format_string(const DVMTypeProvider& type);

/**
 * @param type type to obtain its name in dalvik format
 * @return dalvik format of type as string
 */
std::string get_dalvik_format(const DVMTypeProvider& type);

/**
 * @param type type to obtain its name in canonical format
 * @return canonical format of type as string
 */
std::string_view get_canonical_name(const DVMTypeProvider& type);

/**
 * @param type type to obtain its name in canonical format
 * @return canonical format of type as string
 */
std::string get_canonical_name_string(const DVMTypeProvider& type);

DVMFundamentalProvider * as_fundamental(DVMTypeProvider& type);

DVMClassProvider * as_class(DVMTypeProvider& type);

DVMArrayProvider * as_array(DVMTypeProvider& type);

} // namespace dex
} // namespace shuriken