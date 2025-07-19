//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <vector>
#include <memory>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/internal/providers/dex/custom_types.hpp>
#include <shuriken/internal/io/shurikenstream.hpp>

namespace shuriken {
namespace dex {
class Dex;
class Class;
class DexClassProvider;
class Method;
class DexMethodProvider;
class Field;
class DexFieldProvider;
class FieldID;
class MethodID;

class DVMPrototype;
class DVMPrototypeProvider;

class DexEngine {
private:
    class Impl; // Forward declaration of implementation class
    std::unique_ptr<Impl> pimpl; // The pointer to implementation
    io::ShurikenStream shuriken_stream;
public:
    // Constructor/destructor
    DexEngine(io::ShurikenStream stream, Dex& owner_dex);
    DexEngine(io::ShurikenStream stream, std::string_view dex_path, Dex& owner_dex);
    ~DexEngine();

    /**
     * @brief Parse the DEX file and populate internal data structures
     * @return VoidResult indicating success or error information
     */
    error::VoidResult parse();

    // Move operations
    DexEngine(DexEngine&&) noexcept = default;
    DexEngine& operator=(DexEngine&&) noexcept = default;

    // Disable copy
    DexEngine(const DexEngine&) = delete;
    DexEngine& operator=(const DexEngine&) = delete;

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
     * @param id The string ID index
     * @return String view of the requested string
     */
    std::string_view get_string_by_id(size_t id);

    /**
     * @brief Get the total number of strings in the string pool
     * @return Number of strings in the DEX file
     */
    size_t get_number_of_strings() const;

    /**
     * @brief Get a prototype by its ID from the prototype pool
     * @param id The prototype ID index
     * @return Pointer to the DVMPrototype if found, nullptr otherwise
     */
    DVMPrototype * get_prototype_by_id(size_t id);

    /**
     * @brief Get the total number of prototypes in the prototype pool
     * @return Number of prototypes in the DEX file
     */
    size_t get_number_of_prototypes() const;

    /**
     * @brief Get a type by its ID from the type pool
     * @param id The type ID index
     * @return Pointer to the DVMType if found, nullptr otherwise
     */
    DVMType * get_type_by_id(size_t id);

    /**
     * @brief Get the total number of types in the type pool
     * @return Number of types in the DEX file
     */
    size_t get_number_of_types() const;


    // for classes

    /**
     * @brief Get all classes from the DEX file
     * @return A reference iterator to all the classes from the DEX file
     */
    classes_deref_iterator_t get_classes() const;

    /**
     * @brief Get the total number of classes in the DEX file
     * @return Number of classes defined in the DEX file
     */
    size_t get_number_of_classes() const;

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
     * @brief Get all methods from the DEX file
     * @return A reference iterator to all the methods from the DEX file
     */
    method_deref_iterator_t get_methods() const;


    /**
     * @brief Get all the external methods referenced by the DEX file
     * @return A reference iterator to all the external methods from the DEX file
     */
    external_methods_deref_iterator_t get_external_methods() const;

    /**
     * @brief Get a method ID by its index
     * @param id ID of the MethodID to retrieve inside of the DEX file
     * @return Pointer to a MethodID if exists, or nullptr
     */
    MethodID * get_method_by_id(size_t id);

    /**
     * @brief Get a method ID by its index (const version)
     * @param id ID of the MethodID to retrieve inside of the DEX file
     * @return Const pointer to a MethodID if exists, or nullptr
     */
    const MethodID * get_method_by_id(size_t id) const;


    /**
     * @brief Get the total number of methods in the DEX file
     * @return Number of methods defined in the DEX file
     */
    size_t get_number_of_methods() const;

    /**
    * @brief Find a method by its name and prototype
    * @param name The method name
    * @param prototype The method prototype/signature
    * @return Const pointer to the method if found, nullptr otherwise
    */
    const Method *get_method_by_name_prototype(std::string_view name, std::string_view prototype) const;

    /**
     * @brief Find a method by its name and prototype
     * @param name The method name
     * @param prototype The method prototype/signature
     * @return Pointer to the method if found, nullptr otherwise
     */
    Method *get_method_by_name_prototype(std::string_view name, std::string_view prototype);

    /**
     * @brief Find a method by its full descriptor
     * @param descriptor The complete method descriptor
     * @return Const pointer to the method if found, nullptr otherwise
     */
    const Method *get_method_by_descriptor(std::string_view descriptor) const;

    /**
     * @brief Find a method by its full descriptor
     * @param descriptor The complete method descriptor
     * @return Pointer to the method if found, nullptr otherwise
     */
    Method *get_method_by_descriptor(std::string_view descriptor);

    /**
     * @brief Disassemble a method and populate its instruction list
     * @param method The method provider to disassemble
     */
    void disassemble_method(DexMethodProvider& method);

    /**
     * @brief Get all fields from the DEX file
     * @return A reference iterator to all the fields from the DEX file
     */
    fields_deref_iterator_t get_fields() const;

    /**
     * @brief Get all external fields referenced by the DEX file
     * @return A reference iterator to all the external fields from the DEX file
     */
    external_fields_deref_iterator_t  get_external_fields() const;

    /**
     * @brief Get a field ID by its index
     * @param id The field ID index to retrieve
     * @return Pointer to a FieldID if exists, or nullptr
     */
    FieldID * get_field_by_id(size_t id);

    /**
     * @brief Get a field ID by its index (const version)
     * @param id The field ID index to retrieve
     * @return Const pointer to a FieldID if exists, or nullptr
     */
    const FieldID * get_field_by_id(size_t id) const;

    /**
     * @brief Get the total number of fields in the DEX file
     * @return Number of fields defined in the DEX file
     */
    size_t get_number_of_fields() const;

    /**
   * @brief Find a field by its name
   * @param name The field name
   * @return Const pointer to the field if found, nullptr otherwise
   */
    const Field *get_field_by_name(std::string_view name) const;

    /**
     * @brief Find a field by its name
     * @param name The field name
     * @return Pointer to the field if found, nullptr otherwise
     */
    Field *get_field_by_name(std::string_view name);

    /**
     * @brief Look for methods matching the provided descriptor
     * @param descriptor_regex Regular expression for the method descriptor
     * @return Vector with methods matching the provided descriptor
     */
    std::vector<Method *> found_method_by_regex(std::string_view descriptor_regex);

    /**
     * @brief Look for fields matching the provided descriptor
     * @param descriptor_regex Regular expression for the field descriptor
     * @return Vector with fields matching the provided descriptor
     */
    std::vector<Field *> found_field_by_regex(std::string_view descriptor_regex);
};
}
}