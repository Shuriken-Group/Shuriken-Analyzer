//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>
#include <string_view>
#include <string>

namespace shuriken {
namespace dex {

class DexExternalFieldProvider;

class ExternalField {
private:
    std::reference_wrapper<DexExternalFieldProvider> dex_external_field_provider;
public:
    // constructors & destructors
    ExternalField(DexExternalFieldProvider&);
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