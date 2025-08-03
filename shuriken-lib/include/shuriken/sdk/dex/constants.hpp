//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <unordered_map>

namespace shuriken {
namespace dex {

// ========================================
// DEX File Format Constants
// ========================================

/// @brief Standard endianness marker for DEX files
static const std::uint32_t ENDIAN_CONSTANT = 0x12345678;

/// @brief Reverse endianness marker for DEX files  
static const std::uint32_t REVERSE_ENDIAN_CONSTANT = 0x78563412;

/// @brief Sentinel value indicating no index/invalid reference
static const std::uint32_t NO_INDEX = 0xFFFFFFFF;

/// @brief DEX file magic number prefix
static const std::uint8_t dex_magic[] = {'d', 'e', 'x', '\n'};

/**
 * @namespace shuriken::dex::types
 * @brief Type definitions and enumerations for DEX file analysis
 * 
 * This namespace contains all the type definitions, enumerations, and constants
 * used throughout the DEX analysis framework. It includes access flags, value
 * formats, reference types, and type classifications used in the Dalvik bytecode.
 */
namespace types {
    /// @brief Access flags used in class_def_item,
    /// encoded_field, encoded_method and InnerClass
    /// https://source.android.com/devices/tech/dalvik/dex-format#access-flags
    enum access_flags {
        NONE = 0x0,                        //! No access flags
        ACC_PUBLIC = 0x1,                  //! public type
        ACC_PRIVATE = 0x2,                 //! private type
        ACC_PROTECTED = 0x4,               //! protected type
        ACC_STATIC = 0x8,                  //! static (global) type
        ACC_FINAL = 0x10,                  //! final type (constant)
        ACC_SYNCHRONIZED = 0x20,           //! synchronized
        ACC_VOLATILE = 0x40,               //! Java volatile
        ACC_BRIDGE = 0x40,                 //!
        ACC_TRANSIENT = 0x80,              //!
        ACC_VARARGS = 0x80,                //!
        ACC_NATIVE = 0x100,                //! native type
        ACC_INTERFACE = 0x200,             //! interface type
        ACC_ABSTRACT = 0x400,              //! abstract type
        ACC_STRICT = 0x800,                //!
        ACC_SYNTHETIC = 0x1000,            //!
        ACC_ANNOTATION = 0x2000,           //!
        ACC_ENUM = 0x4000,                 //! enum type
        UNUSED = 0x8000,                   //!
        ACC_CONSTRUCTOR = 0x10000,         //! constructor type
        ACC_DECLARED_SYNCHRONIZED = 0x20000//!
    };

    inline std::string access_flags_to_string(uint32_t flags) {
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

    /**
     * @brief Enumeration for encoded value formats in DEX files
     * 
     * These constants identify the type of data stored in encoded_value structures
     * used in annotations, static field initializers, and other contexts.
     */
    enum value_format : std::uint8_t {
        VALUE_BYTE = 0x0,          //! ubyte[1]
        VALUE_SHORT = 0x2,         //! ubyte[size]
        VALUE_CHAR = 0x3,          //! ubyte[size]
        VALUE_INT = 0x4,           //! ubyte[size]
        VALUE_LONG = 0x6,          //! ubyte[size]
        VALUE_FLOAT = 0x10,        //! ubyte[size]
        VALUE_DOUBLE = 0x11,       //! ubyte[size]
        VALUE_METHOD_TYPE = 0x15,  //! ubyte[size]
        VALUE_METHOD_HANDLE = 0x16,//! ubyte[size]
        VALUE_STRING = 0x17,       //! ubyte[size]
        VALUE_TYPE = 0x18,         //! ubyte[size]
        VALUE_FIELD = 0x19,        //! ubyte[size]
        VALUE_METHOD = 0x1A,       //! ubyte[size]
        VALUE_ENUM = 0x1B,         //! ubyte[size]
        VALUE_ARRAY = 0x1C,        //! EncodedArray
        VALUE_ANNOTATION = 0x1D,   //! EncodedAnnotation
        VALUE_NULL = 0x1E,         //! None
        VALUE_BOOLEAN = 0x1F       //! None
    };

    /**
     * @brief Reference types for cross-reference analysis
     * 
     * These constants identify different types of references between classes,
     * methods, and fields. Used in cross-reference analysis to track how
     * different elements of the DEX file interact with each other.
     */
    enum class ref_type {
        REF_NEW_INSTANCE = 0x22,      // new instance of a class
        REF_CLASS_USAGE = 0x1c,       // class is used somewhere
        REF_INVOKE_VIRTUAL = 0x6e,    // call of a method from a class
        REF_INVOKE_SUPER = 0x6f,      // call of constructor of super class
        REF_INVOKE_DIRECT = 0x70,     // call a method from a class
        REF_INVOKE_STATIC = 0x71,     // call a static method from a class
        REF_INVOKE_INTERFACE = 0x72,  // call an interface method
        // same with ranges
        REF_INVOKE_VIRTUAL_RANGE = 0x74,
        REF_INVOKE_SUPER_RANGE = 0x75,
        REF_INVOKE_DIRECT_RANGE = 0x76,
        REF_INVOKE_STATIC_RANGE = 0x77,
        REF_INVOKE_INTERFACE_RANGE = 0x78
    };

    /**
     * @brief High-level type categories in the DVM type system
     */
    enum class type_e {
        FUNDAMENTAL, //! fundamental type (int, float...)
        CLASS,       //! user defined classes
        ARRAY,       //! array types
    };

    /**
     * @brief Specific fundamental (primitive) types in the DVM
     */
    enum class fundamental_e {
        BOOLEAN,
        BYTE,
        CHAR,
        DOUBLE,
        FLOAT,
        INT,
        LONG,
        SHORT,
        VOID
    };

    /**
     * @brief Method invocation types in Dalvik bytecode
     */
    enum class method_type_e {
        DIRECT_METHOD,  //! Direct method calls (private, static, constructor)
        VIRTUAL_METHOD  //! Virtual method calls (public, protected, package)
    };

    /**
     * @brief Field access types in class definitions
     */
    enum class field_type_e {
        STATIC_FIELD,   //! Class-level static fields
        INSTANCE_FIELD, //! Object instance fields
    };

    const std::unordered_map<fundamental_e, std::string> fundamental_s =
            {
                {fundamental_e::BOOLEAN, "boolean"},
                {fundamental_e::BYTE, "byte"},
                {fundamental_e::CHAR, "char"},
                {fundamental_e::DOUBLE, "double"},
                {fundamental_e::FLOAT, "float"},
                {fundamental_e::INT, "int"},
                {fundamental_e::LONG, "long"},
                {fundamental_e::SHORT, "short"},
                {fundamental_e::VOID, "void"}
            };
} // namespace types

} // namespace dex
} // namespace shuriken