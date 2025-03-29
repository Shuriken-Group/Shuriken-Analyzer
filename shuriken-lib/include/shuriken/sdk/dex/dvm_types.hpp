//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>

namespace shuriken {
namespace dex {
class DVMFundamentalProvider;
class DVMClassProvider;
class DVMArrayProvider;

class DVMFundamental {
private:
    std::reference_wrapper<DVMFundamentalProvider> dvm_fundamental_provider;
public:
    DVMFundamental(DVMFundamentalProvider &);
    ~DVMFundamental() = default;

    DVMFundamental(const DVMFundamental&) = delete;
    DVMFundamental& operator=(const DVMFundamental&) = delete;

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

class DVMClass {
private:
    std::reference_wrapper<DVMClassProvider> dvm_class_provider;
public:
    DVMClass(DVMClassProvider&);
    ~DVMClass() = default;

    DVMClass(const DVMClass&) = delete;
    DVMClass& operator=(const DVMClass&) = delete;

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

class DVMArray {
private:
    std::reference_wrapper<DVMArrayProvider> dvm_array_provider;
public:
    DVMArray(DVMArrayProvider&);
    ~DVMArray() = default;

    DVMArray(const DVMArray&) = delete;
    DVMArray& operator=(const DVMArray&) = delete;

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
    const DVMType* get_base_type() const;
};

/**
 * @param type type to obtain its main type_e
 * @return type_e of provided object
 */
types::type_e get_type(const DVMType& type);

/**
 * @param type type to obtain its name in dalvik format
 * @return dalvik format of type as string_view
 */
std::string_view get_dalvik_format_string(const DVMType& type);

/**
 * @param type type to obtain its name in dalvik format
 * @return dalvik format of type as string
 */
std::string get_dalvik_format(const DVMType& type);

/**
 * @param type type to obtain its name in canonical format
 * @return canonical format of type as string
 */
std::string_view get_canonical_name(const DVMType& type);

/**
 * @param type type to obtain its name in canonical format
 * @return canonical format of type as string
 */
std::string get_canonical_name_string(const DVMType& type);

const DVMFundamental * as_fundamental(const DVMType& type);

const DVMClass * as_class(const DVMType& type);

const DVMArray * as_array(const DVMType& type);

} // namespace dex
} // namespace shuriken