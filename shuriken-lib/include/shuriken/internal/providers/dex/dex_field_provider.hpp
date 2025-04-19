//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <iostream>
#include <vector>

#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>

namespace shuriken {
namespace dex {
class Dex;
class DexEngine;
class Class;

class DexFieldProvider {
private:
    std::reference_wrapper<DexEngine> dex_engine;
    // @brief name of the field without any type, or class name
    std::string field_name;
    // @brief descriptor name like: package/name/class/name->fieldName:type
    std::string descriptor;
    // @brief access flags from the field
    types::access_flags access_flags;
    // @brief type of the field from the generation
    types::field_type_e type;
    // @brief DVMType object that represent the type of the field
    std::reference_wrapper<DVMType> field_type;
    // @brief Pointer to owner class (it can be nullptr)
    std::reference_wrapper<Class>  owner_class;
    // @brief Pointer to owner Dex (it can be nullptr)
    std::reference_wrapper<Dex> owner_dex;
public:
    DexFieldProvider(const std::string& name,
                     DVMType & field_type,
                     types::access_flags access_flags,
                     types::field_type_e type,
                     Class& owner_class,
                     Dex& owner_dex,
                     DexEngine& dex_engine);
    ~DexFieldProvider() = default;

    DexFieldProvider(const DexFieldProvider&) = delete;
    DexFieldProvider& operator=(const DexFieldProvider&) = delete;

    /***
     * @return read-only view from field's name
     */
    std::string_view get_name() const;

    /***
     * @return string with field's name
     */
    std::string get_name_string() const;

    /***
     * @return access flags from the field
     */
    types::access_flags get_field_access_flags() const;

    /***
     * @return get the type from the field `static` or `instance`.
     */
    types::field_type_e get_type() const;

    /***
    * @return const type pointer of the current field
    */
    const DVMType& get_field_type() const;

    /***
     * @return type pointer of the current field
     */
    DVMType& get_field_type();

    /***
     * @return constant pointer to owner class for this Field
     * it can be `nullptr`
     */
    const Class& get_owner_class() const;

    /***
     * @return pointer to owner class for this field
     * it can be `nullptr`
     */
    Class& get_owner_class();

    /***
    * @return constant pointer to dex where the class of this field
    * is
    */
    const Dex& get_owner_dex() const;

    /***
    * @return pointer to dex where the class of this field
    * is
    */
    Dex& get_owner_dex();

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