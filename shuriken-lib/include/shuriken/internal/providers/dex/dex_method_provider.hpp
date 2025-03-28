//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <iostream>
#include <vector>
#include <span>
#include <memory>

#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>


namespace shuriken {
namespace dex {
class Dex;
class DexEngine;
class Class;
class DVMPrototype;

class DexMethodProvider {
private:
    std::reference_wrapper<DexEngine> dex_engine;
    // @brief name of the method
    std::string method_name;
    // @brief access flags from the method
    types::access_flags method_access_flags;
    // @brief pointer to the prototype of the method
    DVMPrototype * method_prototype;
    // @brief Pointer to owner class (it can be nullptr)
    Class * owner_class;
    // @brief Pointer to owner Dex (it can be nullptr)
    Dex * owner_dex;
    // @brief descriptor of the method
    std::string method_descriptor;
    // @brief number of registers used in the bytecode
    std::size_t number_of_registers;
    // @brief span that points to the bytecode
    std::vector<std::uint8_t> bytecode;

    // different xrefs

    std::vector<class_field_idx_t> xref_read_fields;

    std::vector<class_field_idx_t> xref_written_fields;

    std::vector<class_method_idx_t> xref_methods_called;

    std::vector<class_method_idx_t> xref_caller_methods;

    std::vector<class_idx_t> xref_new_instance_classes;

    std::vector<class_idx_t> xref_const_class;

public:
    // constructors & destructors
    DexMethodProvider(const std::string& name,
                      types::access_flags access_flags,
                      DVMPrototype * method_prototype,
                      DexEngine& dex_engine);
    ~DexMethodProvider() = default;

    DexMethodProvider(const DexMethodProvider&) = delete;
    DexMethodProvider& operator=(const DexMethodProvider&) = delete;

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
    types::access_flags get_method_access_flags() const;

    /***
     * @return constant pointer to the prototype of the
     * method. It may return nullptr
     */
    const DVMPrototype* get_method_prototype() const;

    /***
    * @return pointer to the prototype of the
    * method. It may return nullptr
    */
    DVMPrototype* get_method_prototype();

    /***
     * @return constant pointer to owner class for this method
     * it can be `nullptr`
     */
    const Class* get_owner_class() const;

    /***
     * @return pointer to owner class for this method
     * it can be `nullptr`
     */
    Class* get_owner_class();

    /***
     * @return constant pointer to dex where the class of this method
     * is
     */
    const Dex* get_owner_dex() const;

    /***
    * @return pointer to dex where the class of this method
    * is
    */
    Dex* get_owner_dex();

    /***
    * @return a view of method's descriptor
    * package_name/class_name->method_name:prototype
    */
    std::string_view get_descriptor() const;

    /***
     * @return a string of field's descriptor
     * package_name/class_name->method_name:prototype
     */
    std::string get_descriptor_string() const;

    // code item information (more information can be added)

    /**
     * @return get the number of registers used in the current method
     */
    std::uint16_t registers_size() const;

    /**
     * @return return the bytecode that belongs to the method
     */
    std::span<std::uint8_t> get_bytecode() const;
};
}
}