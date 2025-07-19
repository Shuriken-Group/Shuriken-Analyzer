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
class MethodID;
class Field;
class FieldID;
class DVMPrototype;

/**
 * @brief Main entry point for analyzing Android DEX files
 * 
 * This class provides high-level access to all DEX file contents including classes,
 * methods, fields, strings, types, and prototypes. It uses the PIMPL idiom to hide
 * implementation details and manages the DexEngine internally.
 */
class Dex {
private:
    class Impl; // Forward declaration of implementation class
    Impl * pimpl; // The pointer to implementation
public:
    /**
     * @brief Create a Dex object from a file path
     * @param path Path to the DEX file to analyze
     * @return Result containing unique_ptr to Dex object or error
     */
    static error::Result<std::unique_ptr<Dex>> create_from_file(std::string_view path);

    /**
     * @brief Construct a new Dex object
     * @param dex_path Path to the DEX file
     */
    Dex(std::string_view dex_path);
    ~Dex();

    /**
     * @brief Check if the DEX file was successfully initialized
     * @return true if initialized successfully, false otherwise
     */
    bool initialized();

    /**
     * @brief Get the last error that occurred during parsing
     * @return Error object with details about the failure
     */
    error::Error get_last_error();

    /**
     * @brief Get the total number of strings in the string pool
     * @return Number of strings
     */
    size_t get_number_of_strings() const;
    
    /**
     * @brief Get the total number of method prototypes
     * @return Number of prototypes
     */
    size_t get_number_of_prototypes() const;
    
    /**
     * @brief Get the total number of types
     * @return Number of types
     */
    size_t get_number_of_types() const;
    
    /**
     * @brief Get the total number of classes in this DEX file
     * @return Number of classes
     */
    size_t get_number_of_classes() const;
    
    /**
     * @brief Get the total number of methods in this DEX file
     * @return Number of methods
     */
    size_t get_number_of_methods() const;
    
    /**
     * @brief Get the total number of fields in this DEX file
     * @return Number of fields
     */
    size_t get_number_of_fields() const;

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

    /**
     * @brief Get a string from the string pool by its ID
     * @param id Index into the string pool
     * @return String view of the requested string, empty if ID is invalid
     */
    std::string_view get_string_by_id(size_t id);

    /**
     * @brief Get a method prototype by its ID
     * @param id Index into the prototype pool
     * @return Pointer to DVMPrototype if found, nullptr otherwise
     */
    DVMPrototype * get_prototype_by_id(size_t id);

    /**
     * @brief Get a type by its ID
     * @param id Index into the type pool
     * @return Pointer to DVMType if found, nullptr otherwise
     */
    DVMType  * get_type_by_id(size_t id);
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
     * @brief Get all the external methods referenced by the DEX file
     * @return A reference iterator to all the external methods from the DEX file
     */
    external_methods_deref_iterator_t get_external_methods() const;

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
     * @brief Get a method ID by its index
     * @param id ID of the MethodID to retrieve inside of the DEX file
     * @return Pointer to a MethodID if exists, or nullptr
     */
    MethodID * get_method_by_id(size_t id);

    /**
     * @brief Get the Method object corresponding to a MethodID
     * @param method Pointer to the MethodID to lookup
     * @return Pointer to Method if it's an internal method, nullptr otherwise
     */
    Method * get_method_object_by_method_id(MethodID * method);

    /**
     * @brief Get the ExternalMethod object corresponding to a MethodID
     * @param method Pointer to the MethodID to lookup
     * @return Pointer to ExternalMethod if it's an external method, nullptr otherwise
     */
    ExternalMethod * get_external_method_object_by_method_id(MethodID * method);

    /**
     * @return a reference iterator to all the fields from the DEX file
     */
    fields_deref_iterator_t get_fields() const;

    /**
     * @brief Get all external fields referenced by the DEX file
     * @return A reference iterator to all the external fields from the DEX file
     */
    external_fields_deref_iterator_t  get_external_fields() const;

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
     * @brief Get a field ID by its index
     * @param id The field ID index to retrieve
     * @return Pointer to a FieldID if exists, or nullptr
     */
    FieldID * get_field_by_id(size_t id);

    /**
     * @brief Get the Field object corresponding to a FieldID
     * @param field Pointer to the FieldID to lookup
     * @return Pointer to Field if it's an internal field, nullptr otherwise
     */
    Field * get_field_object_by_field_id(FieldID * field);

    /**
     * @brief Get the ExternalField object corresponding to a FieldID
     * @param field Pointer to the FieldID to lookup
     * @return Pointer to ExternalField if it's an external field, nullptr otherwise
     */
    ExternalField * get_external_field_object_by_field_id(FieldID * field);

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