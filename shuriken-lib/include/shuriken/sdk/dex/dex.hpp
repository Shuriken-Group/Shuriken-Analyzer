//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/common/error.hpp>

namespace shuriken {
namespace dex {
class DexEngine;
class Class;
class Method;
class Field;

class Dex {
private:
    class Impl; // Forward declaration of implementation class
    Impl * pimpl; // The pointer to implementation
public:
    // static methods
    static error::Result<std::unique_ptr<Dex>> create_from_file(std::string_view path);

    // Constructors & Destructors
    Dex(std::string_view dex_path);
    ~Dex();

    bool initialized();

    error::Error get_last_error();

    /**
     * @brief Get the path of the DEX file as a string_view
     * @return String view of the DEX file path
     */
    std::string_view get_dex_path() const;

    /**
     * @brief Get the path of the DEX file as a string
     * @return String copy of the DEX file path
     */
    std::string get_dex_path_string() const;

    /**
     * @brief Get the filename of the DEX file as a string_view
     * @return String view of the DEX filename
     */
    std::string_view get_dex_name() const;

    /**
     * @brief Get the filename of the DEX file as a string
     * @return String copy of the DEX filename
     */
    std::string get_dex_name_string() const;

    // for classes

    /**
     * @brief Get all classes from the DEX file
     * @return A reference iterator to all the classes from the DEX file
     */
    classes_deref_iterator_t get_classes() const;

    /**
     * @brief Find a class by its package name and class name
     * @param package_name The package part of the class name
     * @param name The simple name of the class
     * @return Const pointer to the class if found, nullptr otherwise
     */
    const Class *get_class_by_package_name_and_name(std::string_view package_name, std::string_view name) const;

    /**
     * @brief Find a class by its package name and class name
     * @param package_name The package part of the class name
     * @param name The simple name of the class
     * @return Pointer to the class if found, nullptr otherwise
     */
    Class *get_class_by_package_name_and_name(std::string_view package_name, std::string_view name);

    /**
     * @brief Find a class by its full descriptor
     * @param descriptor The complete class descriptor (e.g., "Ljava/lang/String;")
     * @return Const pointer to the class if found, nullptr otherwise
     */
    const Class *get_class_by_descriptor(std::string_view descriptor) const;

    /**
     * @brief Find a class by its full descriptor
     * @param descriptor The complete class descriptor (e.g., "Ljava/lang/String;")
     * @return Pointer to the class if found, nullptr otherwise
     */
    Class *get_class_by_descriptor(std::string_view descriptor);

    /**
     * @brief Find classes matching a regular expression pattern
     * @param descriptor_regex Regular expression to match against class descriptors
     * @return Vector of pointers to matching Class objects
     */
    std::vector<Class *> find_classes_by_regex(std::string_view descriptor_regex);


    /**
     * @return a reference iterator to all the methods from the DEX file
     */
    method_deref_iterator_t get_methods() const;

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
     * @return a reference iterator to all the fields from the DEX file
     */
    fields_deref_iterator_t get_fields() const;

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
}
}