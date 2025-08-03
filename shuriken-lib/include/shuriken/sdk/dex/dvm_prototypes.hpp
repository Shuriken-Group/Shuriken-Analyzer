//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include "shuriken/sdk/dex/custom_types.hpp"

#include <string_view>
#include <memory>
#include <string>

namespace shuriken {
namespace dex {

/**
 * @brief Represents a method prototype/signature from a DEX file
 * 
 * A DVMPrototype describes the signature of a method, including its return type
 * and parameter types. It provides both the "shorty" representation (compact form)
 * and the full descriptor. This class is fundamental for method identification
 * and type analysis in DEX files.
 */
class DVMPrototype {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new DVMPrototype object
     * @param impl Pointer to the implementation containing prototype data
     */
    DVMPrototype(Impl*);
    ~DVMPrototype() = default;

    /**
     * @brief Get the shorty descriptor of the prototype
     * 
     * The shorty is a compact representation of the method signature using
     * single characters: V (void), Z (boolean), B (byte), S (short), C (char),
     * I (int), J (long), F (float), D (double), L (object/array).
     * 
     * @return String view of the shorty descriptor (e.g., "VIL" for void method(int, Object))
     */
    std::string_view get_shorty_idx() const;

    /**
     * @brief Get the shorty descriptor as a string copy
     * @return String copy of the shorty descriptor
     */
    std::string get_shorty_idx_string() const;

    /**
     * @brief Get the return type of this method prototype
     * @return Const reference to the DVMType representing the return type
     */
    const DVMType& get_return_type() const;

    /**
     * @brief Get the return type of this method prototype
     * @return Reference to the DVMType representing the return type
     */
    DVMType & get_return_type();

    /**
     * @brief Get all parameter types for this method prototype
     * @return Iterator range over the parameter types in order
     */
    dvmtypes_list_deref_iterator_t get_parameters();

    /**
     * @brief Get the full method descriptor
     * 
     * The descriptor contains the complete method signature including parameter
     * types and return type in the format: (param1param2...)returntype
     * Example: "(ILjava/lang/String;)V" for method(int, String) returning void
     * 
     * @return String view of the method descriptor
     */
    std::string_view get_descriptor();

    /**
     * @brief Get the full method descriptor as a string copy
     * @return String copy of the method descriptor
     */
    std::string get_descriptor_string();
};

}
}