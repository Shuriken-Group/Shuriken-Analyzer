//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>

namespace shuriken {
namespace dex {

class DexEngine;

class DexExternalFieldProvider {
private:
    std::reference_wrapper<DexEngine> dex_engine;
    // @brief name of the field without any type, or class name
    std::string field_name;
    // @brief descriptor name like: package/name/class/name->fieldName:type
    std::string descriptor;
    // @brief class name
    std::string class_name;
public:
    DexExternalFieldProvider(
            std::string_view field_name,
            std::string_view descriptor,
            std::string_view class_name,
            DexEngine& dex_engine
            );

    ~DexExternalFieldProvider() = default;
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

}
}