//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <string>
#include <string_view>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/common/iterator_range.hpp>

namespace shuriken {
namespace dex {

class DexClassProvider {
private:
    // @brief name of the class
    std::string name;
    // @brief name of the package
    std::string package_name;
    // @brief dalvik name format of the class name
    std::string dalvik_format;
    // @brief canonical name format of the class name
    std::string canonical_format;
    // @brief extended class
    class_external_class_t extended_class;
    // @brief possible implemented classes
    std::vector<class_external_class_t> implemented_classes;
    // @brief Methods from the class
    std::vector<method_t> methods;
    // @brief fields from the class
    std::vector<field_t> fields;
public:
    DexClassProvider(std::string name, std::string package_name,
                     externalclass_t extended_class, std::vector<class_external_class_t>& implemented_classes,
                     std::vector<method_t>& methods, std::vector<field_t>& fields);
    ~DexClassProvider() = default;

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

};

} // namespace dex
} // namespace shuriken