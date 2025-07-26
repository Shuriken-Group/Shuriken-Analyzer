//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>
#include <string>
#include <string_view>

#include <shuriken/sdk/common/iterator_range.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/disassembly_constants.hpp>
#include <shuriken/sdk/dex/instruction.hpp>

namespace shuriken {
namespace dex {

class Dex;
class Class;
class DVMPrototype;
class Instruction;

/**
 * @brief Represents a method from a DEX file
 * 
 * This class provides access to method metadata, bytecode, instructions, and cross-references.
 * It acts as a lightweight wrapper around DexMethodProvider containing the actual data.
 */
class Method {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new Method object
     * @param provider Reference to the method provider containing the actual data
     */
    Method(Impl*);
    ~Method() = default;

    Method(const Method&) = delete;
    Method& operator=(const Method&) = delete;

    // information from the method

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
     * @return a string reference with the access flags expressed as
     * FLAG1|FLAG2|...
     */
    std::string_view get_method_access_flags_str();

    /***
     * @return constant pointer to the prototype of the
     * method. It may return nullptr
     */
     const DVMPrototype& get_method_prototype() const;

    /***
    * @return pointer to the prototype of the
    * method. It may return nullptr
    */
    DVMPrototype& get_method_prototype();

    /***
     * @return get the type of the method.
     */
    types::method_type_e get_method_type() const;

    /***
     * @return constant pointer to owner class for this method
     * it can be `nullptr`
     */
    const Class& get_owner_class() const;

    /***
     * @return pointer to owner class for this method
     * it can be `nullptr`
     */
    Class& get_owner_class();

    /***
     * @return constant pointer to dex where the class of this method
     * is
     */
    const Dex& get_owner_dex() const;

    /***
    * @return pointer to dex where the class of this method
    * is
    */
    Dex& get_owner_dex();

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
     * @return return the op_codes that belongs to the method
     */
    std::span<std::uint8_t> get_bytecode();


    // Disassembler information
    /**
     * @brief Get the disassembled instructions for this method
     * @return Reference to list of instruction references
     */
    std::list<std::reference_wrapper<Instruction>> & get_method_instructions();

    /**
     * @brief Get exception handling information for this method
     * @return Reference to exception data structure
     */
    disassembler::exceptions_data_t & get_exceptions();

     // xrefs information
     /**
      * @return iterator to a structure of type std::tuple<Class*,Field*,uint64_t>,
      * indicating the Class->Field that is read, and the index in the method
      * where it is read.
      */
     iterator_range<span_class_field_idx_iterator_t>
     get_xref_read_fields_in_method();

    /**
     * @return iterator to a structure of type std::tuple<Class*,Field*,uint64_t>,
     * indicating the Class->Field that is written, and the index in the method
     * where it is written.
     */
     iterator_range<span_class_field_idx_iterator_t>
     get_xref_written_fields_in_method();

     /**
      * @return iterator to a structure of type std::tuple<Class*,Method*,uint64_t>,
      * it represents the methods that call the current method.
      */
     iterator_range<span_class_method_idx_iterator_t>
     get_xref_methods_called();

     /**
      * @return iterator to a structure of type std::tuple<Class*,Method*,uint64_t>
      * it represents the methods the current method calls.
      */
    iterator_range<span_class_method_idx_iterator_t>
    get_xref_caller_methods();

    /**
     * @return iterator to a structure of type std::tuple<Class*, uint64_t>
     * it represents the instantiated classes in the method
     */
    iterator_range<span_class_idx_iterator_t>
    get_xref_new_instance_classes();

    /**
     * @return iterator to a structure of type std::tuple<Class*, uint64_t>
     * it represents the use of a constant class in the method
     */
    iterator_range<span_class_idx_iterator_t>
    get_xref_const_class();
};

} // namespace dex
} // namespace shuriken