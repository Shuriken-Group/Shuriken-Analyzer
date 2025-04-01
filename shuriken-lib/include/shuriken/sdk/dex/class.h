//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include <string>
#include <string_view>

#include <shuriken/sdk/common/iterator_range.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>

namespace shuriken {
namespace dex {
class DexClassProvider;
class Dex;
class Method;
class Field;

class Class {
private:
    std::reference_wrapper<DexClassProvider> dex_class_provider;
public:
    // constructors & destructors
    Class(DexClassProvider &);

    ~Class() = default;

    Class(const Class &) = delete;

    Class &operator=(const Class &) = delete;

    // information from the class

    /***
     * @return read-only view from class' name
     */
    std::string_view get_name() const;

    /***
     * @return string with class' name
     */
    std::string get_name_string() const;

    /**
     * @return name of the package from the class
     */
    std::string_view get_package_name() const;

    /**
     * @return name of the package as string
     */
    std::string get_package_name_string() const;

    /**
     * @return name of the class in dalvik format as
     * package/name->className
     */
    std::string_view get_dalvik_name() const;

    /**
    * @return name of the class in dalvik format as
    * package/name->className as string
    */
    std::string get_dalvik_name_string() const;

    /**
     * @return name of the class in canonical format as
     * package.name.ClassName
     */
    std::string_view get_canonical_name() const;

    /**
    * @return name of the class in canonical format as
     * package.name.ClassName as string
    */
    std::string get_canonical_name_string() const;


    /**
     * @return A std::variant representing both possibilities:
     *         - Class* in case it extends a class inside the DEX
     *         - ExternalClass* in case it extends a class outside the DEX
     *         Returns empty variant if the class doesn't extend any class
     */
    class_external_class_t get_extended_class();

    /**
     * @return The number of interfaces implemented by this class
     */
    std::size_t get_number_of_implemented_classes();

    /**
     * @return An iterator range over the interfaces implemented by this class
     *         Each element can be either a Class* (internal) or ExternalClass* (external)
     */
    iterator_range<span_class_external_class_iterator_t> get_implemented_classes();



    // A few interesting getters for fields
    // and methods belonging to the current class

    /**
     * @return number of methods in the class
     */
    std::size_t get_number_of_methods() const;

    /**
     * @return An iterator range of all methods in this class
     */
    method_deref_iterator_t get_methods();

    /**
     * Find a method by its name and prototype
     * @param name The method name
     * @param prototype The method prototype/signature
     * @return Const pointer to the method if found, nullptr otherwise
     */
    const Method *get_method_by_name_prototype(std::string_view name, std::string_view prototype) const;

    /**
     * Find a method by its name and prototype
     * @param name The method name
     * @param prototype The method prototype/signature
     * @return Pointer to the method if found, nullptr otherwise
     */
    Method *get_method_by_name_prototype(std::string_view name, std::string_view prototype);

    /**
     * Find a method by its full descriptor
     * @param descriptor The complete method descriptor
     * @return Const pointer to the method if found, nullptr otherwise
     */
    const Method *get_method_by_descriptor(std::string_view descriptor) const;

    /**
     * Find a method by its full descriptor
     * @param descriptor The complete method descriptor
     * @return Pointer to the method if found, nullptr otherwise
     */
    Method *get_method_by_descriptor(std::string_view descriptor);

    /**
     * @return number of fields in the class
     */
    std::size_t get_number_of_fields() const;

    /**
     * @return An iterator range of all fields in this class
     */
    fields_deref_iterator_t get_fields();

    /**
     * Find a field by its name
     * @param name The field name
     * @return Const pointer to the field if found, nullptr otherwise
     */
    const Field *get_field_by_name(std::string_view name) const;

    /**
     * Find a field by its name
     * @param name The field name
     * @return Pointer to the field if found, nullptr otherwise
     */
    Field *get_field_by_name(std::string_view name);

    /**
     * Look for methods matching the provided descriptor.
     * @param descriptor_regex regular expression for the method descriptor
     * @return vector with methods matching the provided descriptor
     */
    std::vector<Method *> found_method_by_regex(std::string_view descriptor_regex);

    /**
     * Look for fields matching the provided descriptor.
     * @param descriptor_regex regular expression for the field descriptor
     * @return vector with fields matching the provided descriptor
     */
    std::vector<Field *> found_field_by_regex(std::string_view descriptor_regex);

    // xrefs information
    /**
     * @return An iterator range over all classes referenced by this class
     *         Each element is a mapping from the referenced class to a set
     *         of tuples containing (reference type, calling method, instruction offset)
     */
    iterator_range<classxref_iterator_t> get_outgoing_class_references();

    /**
     * @return An iterator range over all classes that reference this class
     *         Each element is a mapping from the referencing class to a set
     *         of tuples containing (reference type, calling method, instruction offset)
     */
    iterator_range<classxref_iterator_t> get_incoming_class_references();

    /**
     * @return new instance of this class in different methods.
     */
    iterator_range<span_method_idx_iterator_t>
    get_xref_new_instance_methods();

    /**
     * @return use of this class as a constant class in different methods.
     */
    iterator_range<span_method_idx_iterator_t>
    get_xref_const_class();
};

} // namespace dex
} // namespace shuriken