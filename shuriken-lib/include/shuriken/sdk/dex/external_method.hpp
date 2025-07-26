//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <memory>
#include <string_view>
#include <string>

namespace shuriken {
namespace dex {

/**
 * @brief Represents a method reference from another DEX file
 * 
 * ExternalMethod objects are created for method references that point to classes
 * not defined in the current DEX file. They provide basic metadata but no
 * implementation details since the actual method code is in another file.
 */
class ExternalMethod {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new ExternalMethod object
     * @param provider Reference to the external method provider containing the data
     */
    ExternalMethod(Impl*);
    ~ExternalMethod() = default;

    /**
    * @brief Get the name of the class that owns this external method as a string_view
    * @return String view of the owner class name
    */
    std::string_view get_class_name() const;

    /**
     * @brief Get the name of the class that owns this external method as a string
     * @return String copy of the owner class name
     */
    std::string get_class_name_string() const;

    /**
     * @brief Get the name of this external method as a string_view
     * @return String view of the method name
     */
    std::string_view get_name() const;

    /**
     * @brief Get the name of this external method as a string
     * @return String copy of the method name
     */
    std::string get_name_string() const;

    /***
     * @return a view of method's descriptor
     * package_name/class_name->method_name(params)retType
     */
    std::string_view get_descriptor() const;

    /***
     * @return a string of method's descriptor
     * package_name/class_name->method_name(params)retType
     */
    std::string get_descriptor_string() const;
};


} // namespace dex
} // namespace shuriken