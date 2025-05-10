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

namespace shuriken {
namespace dex {

class DexMethodProvider;
class Dex;
class Class;
class DVMPrototype;
class InstructionProvider;

class Method {
private:
    std::reference_wrapper<DexMethodProvider> dex_method_provider;
public:
    // constructors and destructors
    Method(DexMethodProvider&);
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
     * @return return the bytecode that belongs to the method
     */
    std::span<const std::uint8_t> get_bytecode() const;


    // Disassembler information

    /**
     * @return number of instructions in the method
     */
     size_t get_number_of_instructions() const;

    /**
     * @return iterator to a list of instructions
     */
    instruction_list_deref_iterator_t get_instructions() const;

    /**
     * @return get a span object with all the instructions
     */
     const instruction_list_t get_instructions_container() const;

    /**
     * @param idx index of the instruction to retrieve
     * @return pointer to instruction in an specific idx, may return null
     */
     InstructionProvider * get_instruction_at(std::uint64_t idx);


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