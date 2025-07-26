//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <string_view>
#include <string>
#include <memory>

namespace shuriken {
namespace dex {

/**
 * @brief Represents a field reference from another DEX file
 * 
 * ExternalField objects are created for field references that point to classes
 * not defined in the current DEX file. They provide basic metadata but no
 * implementation details since the actual field definition is in another file.
 */
class ExternalField {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new ExternalField object
     * @param external Reference to the external field provider containing the data
     */
    ExternalField(Impl*);
    ~ExternalField() = default;

    /**
    * @brief Get the name of the class that owns this external field as a string_view
    * @return String view of the owner class name
    */
    std::string_view get_class_name() const;

    /**
     * @brief Get the name of the class that owns this external field as a string
     * @return String copy of the owner class name
     */
    std::string get_class_name_string() const;

    /**
     * @brief Get the name of this external field as a string_view
     * @return String view of the field name
     */
    std::string_view get_name() const;

    /**
     * @brief Get the name of this external field as a string
     * @return String copy of the field name
     */
    std::string get_name_string() const;

    /***
     * @return a view of field's descriptor
     * package_name/class_name->field_name:type
     */
    std::string_view get_descriptor() const;

    /***
     * @return a string of field's descriptor
     * package_name/class_name->field_name:type
     */
    std::string get_descriptor_string() const;
};


} // namespace dex
} // namespace shuriken