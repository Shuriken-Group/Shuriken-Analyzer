//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>

#include <memory>

namespace shuriken {
namespace dex {

/**
 * @brief Represents a fundamental (primitive) type in the Dalvik Virtual Machine
 * 
 * DVMFundamental represents primitive types like int, long, boolean, etc.
 * These are the basic building blocks of the type system in DEX files.
 * Examples include: void (V), boolean (Z), byte (B), char (C), short (S),
 * int (I), long (J), float (F), double (D).
 */
class DVMFundamental {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new DVMFundamental object
     * @param impl Pointer to the implementation containing type data
     */
    DVMFundamental(Impl*);
    ~DVMFundamental() = default;

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

/**
 * @brief Represents a class/object type in the Dalvik Virtual Machine
 * 
 * DVMClass represents reference types including classes, interfaces, and enums.
 * These types are represented in Dalvik format as Lpackage/name/ClassName;
 * and in canonical format as package.name.ClassName.
 */
class DVMClass {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new DVMClass object
     * @param impl Pointer to the implementation containing class type data
     */
    DVMClass(Impl*);
    ~DVMClass() = default;

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

/**
 * @brief Represents an array type in the Dalvik Virtual Machine
 * 
 * DVMArray represents array types of any dimension and base type.
 * Arrays in Dalvik are represented with '[' prefixes indicating depth,
 * followed by the element type descriptor. Examples:
 * - [I (int array)
 * - [[Ljava/lang/String; (2D String array)
 */
class DVMArray {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new DVMArray object
     * @param impl Pointer to the implementation containing array type data
     */
    DVMArray(Impl*);
    ~DVMArray() = default;

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

// ========================================
// Type System Utility Functions
// ========================================

/**
 * @brief Get the general type category of a DVMType
 * @param type The type to examine
 * @return The type_e enum indicating whether it's FUNDAMENTAL, CLASS, or ARRAY
 */
types::type_e get_type(const DVMType& type);

/**
 * @brief Get the Dalvik format representation as a string view
 * @param type The type to format
 * @return String view of the Dalvik format (e.g., "I", "Ljava/lang/String;", "[I")
 */
std::string_view get_dalvik_format_string(const DVMType& type);

/**
 * @brief Get the Dalvik format representation as a string copy
 * @param type The type to format
 * @return String copy of the Dalvik format
 */
std::string get_dalvik_format(const DVMType& type);

/**
 * @brief Get the canonical Java name as a string view
 * @param type The type to format
 * @return String view of the canonical name (e.g., "int", "java.lang.String", "int[]")
 */
std::string_view get_canonical_name(const DVMType& type);

/**
 * @brief Get the canonical Java name as a string copy
 * @param type The type to format
 * @return String copy of the canonical name
 */
std::string get_canonical_name_string(const DVMType& type);

// ========================================
// Type Casting Functions
// ========================================

/**
 * @brief Cast a DVMType to DVMFundamental if it's a fundamental type
 * @param type The type to cast
 * @return Pointer to DVMFundamental if successful, nullptr otherwise
 */
const DVMFundamental * as_fundamental(const DVMType& type);

/**
 * @brief Cast a DVMType to DVMClass if it's a class type
 * @param type The type to cast
 * @return Pointer to DVMClass if successful, nullptr otherwise
 */
const DVMClass * as_class(const DVMType& type);

/**
 * @brief Cast a DVMType to DVMArray if it's an array type
 * @param type The type to cast
 * @return Pointer to DVMArray if successful, nullptr otherwise
 */
const DVMArray * as_array(const DVMType& type);

} // namespace dex
} // namespace shuriken