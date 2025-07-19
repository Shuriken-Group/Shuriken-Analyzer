//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <filesystem>
#include <algorithm>
#include <regex>
#include <unordered_map>

#include <shuriken/internal/engine/dex/dex_engine.hpp>

#include "shuriken/sdk/dex/dex.hpp"
#include "shuriken/sdk/dex/class.hpp"
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/sdk/dex/external_method.hpp"
#include "shuriken/sdk/dex/field.hpp"
#include "shuriken/sdk/dex/external_field.hpp"
#include "shuriken/internal/providers/dex/dex_class_provider.hpp"
#include "shuriken/internal/providers/dex/dex_method_provider.hpp"
#include "shuriken/internal/providers/dex/dex_external_method_provider.hpp"
#include "shuriken/internal/providers/dex/dex_field_provider.hpp"
#include "shuriken/internal/providers/dex/dex_external_field_provider.hpp"
#include "shuriken/internal/providers/dex/dvm_prototypes_provider.hpp"

#include "shuriken/internal/engine/dex/parser/parser.hpp"
#include "shuriken/internal/engine/dex/disassembler/internal_disassembler.hpp"

using namespace shuriken::dex;

namespace {
    /**
     * @brief Split a class descriptor into package and class name components
     * @param descriptor Class descriptor in Dalvik format (e.g., "Ljava/lang/String;")
     * @return Tuple containing package name and class name
     */
    std::tuple<std::string, std::string> split_class_descriptor(std::string_view descriptor) {
        // Check if the descriptor is valid (starts with 'L' and ends with ';')
        if (descriptor.empty() || descriptor[0] != 'L' || descriptor.back() != ';') {
            return {"", ""}; // Return empty strings for invalid descriptor
        }

        // Remove the 'L' prefix and ';' suffix
        std::string_view type_name = descriptor.substr(1, descriptor.size() - 2);

        // Find the last '/' to separate package from class name
        size_t last_slash = type_name.rfind('/');

        if (last_slash == std::string_view::npos) {
            // No package, just a class name
            return {"", std::string(type_name)};
        } else {
            // Extract package and class name
            std::string package(type_name.substr(0, last_slash));
            std::string class_name(type_name.substr(last_slash + 1));
            return {package, class_name};
        }
    }
}

/**
 * @brief Pimpl implementation class for DexEngine
 *
 * This class contains all the actual data and implementation details,
 * keeping the public header clean and minimal. It uses the Provider pattern
 * where SDK objects are lightweight handles pointing to provider implementations.
 */
class DexEngine::Impl {
public:
    // ========================================
    // Core Engine Components
    // ========================================

    /// DEX file parser for handling binary format
    Parser parser;

    /// Reference to the owning Dex object (SDK interface)
    std::reference_wrapper<Dex> owner_dex;

    /// Disassembler service for converting bytecode to instruction objects
    std::unique_ptr<InternalDisassembler> disassembler;

    // ========================================
    // File Information
    // ========================================

    /// Full path to the DEX file on disk
    std::string dex_path;

    /// Just the filename portion of the DEX file
    std::string dex_name;

    // ========================================
    // DEX Data Pools (extracted from file)
    // ========================================

    /// String pool from the DEX file containing all string literals
    std::vector<std::string> strings_pool;

    // ========================================
    // Object Ownership - Classes
    // ========================================

    /// SDK interface objects for classes (user-facing lightweight handles)
    std::vector<std::unique_ptr<Class>> sdk_classes;

    /// Provider objects containing actual class data and implementation
    std::vector<std::unique_ptr<DexClassProvider>> dex_class_providers;

    // ========================================
    // Object Ownership - Methods
    // ========================================

    /// SDK interface objects for methods (user-facing lightweight handles)
    std::vector<std::unique_ptr<Method>> sdk_methods;

    /// Provider objects containing actual method data and implementation
    std::vector<std::unique_ptr<DexMethodProvider>> dex_methods_providers;

    /// SDK interface objects for external methods (references to other DEX files)
    std::vector<std::unique_ptr<ExternalMethod>> sdk_external_methods;

    /// Provider objects for external method implementations
    std::vector<std::unique_ptr<DexExternalMethodProvider>> dex_external_methods_providers;

    // ========================================
    // Object Ownership - Fields
    // ========================================

    /// SDK interface objects for fields (user-facing lightweight handles)
    std::vector<std::unique_ptr<Field>> sdk_fields;

    /// Provider objects containing actual field data and implementation
    std::vector<std::unique_ptr<DexFieldProvider>> dex_fields_providers;

    /// SDK interface objects for external fields (references to other DEX files)
    std::vector<std::unique_ptr<ExternalField>> sdk_external_fields;

    /// Provider objects for external field implementations
    std::vector<std::unique_ptr<DexExternalFieldProvider>> dex_external_fields_providers;

    // ========================================
    // Object Ownership - Type System
    // ========================================

    /// SDK interface objects for method prototypes/signatures
    std::vector<std::unique_ptr<DVMPrototype>> sdk_prototypes;

    /// Provider objects for prototype implementations
    std::vector<std::unique_ptr<DVMPrototypeProvider>> dex_prototypes_providers;

    /// SDK interface objects for DVM types
    std::vector<std::unique_ptr<DVMType>> sdk_dvmtypes;

    /// Provider objects for type implementations
    std::vector<std::unique_ptr<DVMTypeProvider>> dex_type_providers;

    // ========================================
    // Reference Caches (for fast iteration without unique_ptr dereferencing)
    // ========================================

    /// Fast access reference wrappers for classes
    std::vector<std::reference_wrapper<Class>> ref_sdk_classes;
    std::vector<std::reference_wrapper<DexClassProvider>> ref_dex_class_providers;

    /// Fast access reference wrappers for methods
    std::vector<std::reference_wrapper<Method>> ref_sdk_methods;
    std::vector<std::reference_wrapper<DexMethodProvider>> ref_dex_methods_providers;

    /// Fast access reference wrappers for external methods
    std::vector<std::reference_wrapper<ExternalMethod>> ref_sdk_external_methods;
    std::vector<std::reference_wrapper<DexExternalMethodProvider>> ref_dex_external_methods_providers;

    /// Fast access reference wrappers for other object types
    std::vector<std::reference_wrapper<Field>> ref_sdk_fields;
    std::vector<std::reference_wrapper<DexFieldProvider>> ref_dex_fields_providers;

    /// Fast access reference wrappers for external fields
    std::vector<std::reference_wrapper<ExternalField>> ref_sdk_externals_fields;
    std::vector<std::reference_wrapper<DexExternalFieldProvider>> ref_dex_externals_fields_providers;


    std::vector<std::reference_wrapper<DVMPrototype>> ref_sdk_prototypes;
    std::vector<std::reference_wrapper<DVMPrototypeProvider>> ref_dex_prototypes_providers;
    std::vector<std::reference_wrapper<DVMType>> ref_sdk_dvmtypes;
    std::vector<std::reference_wrapper<DVMTypeProvider>> ref_dex_type_providers;

    // ========================================
    // Method ID Mappings (for O(1) lookups from parser data to our objects)
    // ========================================

    /// Maps MethodID from DEX file to our SDK Method objects
    std::unordered_map<MethodID *, Method *> method_id_method;

    /// Maps MethodID from DEX file to our Method provider implementations
    std::unordered_map<MethodID *, DexMethodProvider *> method_id_method_provider;

    /// Maps MethodID from DEX file to our SDK ExternalMethod objects
    std::unordered_map<MethodID *, ExternalMethod *> method_id_external_method;

    /// Maps MethodID from DEX file to our ExternalMethod provider implementations
    std::unordered_map<MethodID *, DexExternalMethodProvider *> method_id_external_method_provider;


    // ========================================
    // Field ID Mappings (for O(1) lookups from parser data to our objects)
    // ========================================

    std::unordered_map<FieldID *, Field *> field_id_field;

    std::unordered_map<FieldID *, DexFieldProvider *> field_id_field_provider;

    std::unordered_map<FieldID *, ExternalField *> field_id_external_field;

    std::unordered_map<FieldID *, DexExternalFieldProvider *> field_id_external_field_provider;

    // ========================================
    // Constructor
    // ========================================

    Impl(Dex &owner_dex) : owner_dex(owner_dex) {}

    ~Impl() = default;

    // ========================================
    // Object Factory Methods (maintain SDK-Provider relationships)
    // ========================================

    /**
     * @brief Store a new class and its provider, maintaining all reference collections
     * @param new_class The provider containing class implementation
     */
    void save_class(std::unique_ptr<DexClassProvider> &new_class) {
        auto new_sdk_class = std::make_unique<Class>(*new_class);
        this->dex_class_providers.push_back(std::move(new_class));
        this->ref_dex_class_providers.push_back(std::ref(*this->dex_class_providers.back().get()));
        this->sdk_classes.push_back(std::move(new_sdk_class));
        this->ref_sdk_classes.push_back(std::ref(*this->sdk_classes.back().get()));
    }

    /**
     * @brief Store a new method and its provider, maintaining all reference collections and mappings
     * @param method_provider The provider containing method implementation
     * @param method_id The MethodID from DEX file for creating lookup mappings
     */
    void save_method(std::unique_ptr<DexMethodProvider> &method_provider, MethodID *method_id) {
        auto method_sdk = std::make_unique<Method>(*method_provider.get());
        this->dex_methods_providers.push_back(std::move(method_provider));
        this->ref_dex_methods_providers.push_back(*this->dex_methods_providers.back());
        this->method_id_method_provider[method_id] = &(this->ref_dex_methods_providers.back().get());

        this->sdk_methods.push_back(std::move(method_sdk));
        this->ref_sdk_methods.push_back(*this->sdk_methods.back());
        this->method_id_method[method_id] = &(this->ref_sdk_methods.back().get());

        this->dex_class_providers.back()->add_method(this->ref_sdk_methods.back());
    }

    void save_external_method(MethodID *method_id, DexEngine &engine) {
        auto descriptor = ::get_dalvik_format(method_id->get_class()) + "->" + method_id->get_name_string() +
                          method_id->get_prototype().get_descriptor_string();
        auto external_method_provider = std::make_unique<DexExternalMethodProvider>(
                method_id->get_name_string(),
                descriptor,
                ::get_dalvik_format_string(method_id->get_class()),
                engine
        );
        auto external_method = std::make_unique<ExternalMethod>(*external_method_provider.get());

        this->dex_external_methods_providers.push_back(std::move(external_method_provider));
        this->ref_dex_external_methods_providers.push_back(*this->dex_external_methods_providers.back());
        this->method_id_external_method_provider[method_id] = &(this->ref_dex_external_methods_providers.back().get());

        this->sdk_external_methods.push_back(std::move(external_method));
        this->ref_sdk_external_methods.push_back(*this->sdk_external_methods.back());
        this->method_id_external_method[method_id] = &(this->ref_sdk_external_methods.back().get());
    }

    /**
     * @brief Get method by MethodID, creating external method if needed
     * @param method_id The method identifier from DEX file
     * @param engine Reference to engine for creating external methods
     * @return Variant containing either Method or ExternalMethod
     *
     * This function implements lazy loading:
     * 1. Checks internal methods first
     * 2. Checks cached external methods
     * 3. Creates external method on-demand if not found
     */
    method_external_method_t get_method_by_method_id(MethodID *method_id, DexEngine &engine) {
        auto it = method_id_method.find(method_id);

        if (it != method_id_method.end()) return *it->second;

        auto it_e = this->method_id_external_method.find(method_id);

        if (it_e != this->method_id_external_method.end()) return *it_e->second;

        save_external_method(method_id, engine);

        return *this->method_id_external_method[method_id];
    }

    /**
     * @brief Store a new field and its provider, maintaining all reference collections
     * @param field_provider The provider containing field implementation
     */
    void save_field(std::unique_ptr<DexFieldProvider> &field_provider, FieldID *field_id) {
        auto field_sdk = std::make_unique<Field>(*field_provider.get());

        this->dex_fields_providers.push_back(std::move(field_provider));
        this->ref_dex_fields_providers.push_back(*this->dex_fields_providers.back());
        this->field_id_field_provider[field_id] = &(this->ref_dex_fields_providers.back().get());

        this->sdk_fields.push_back(std::move(field_sdk));
        this->ref_sdk_fields.push_back(*this->sdk_fields.back());
        this->field_id_field[field_id] = &(this->ref_sdk_fields.back().get());

        this->dex_class_providers.back()->add_field(this->ref_sdk_fields.back());
    }

    void save_external_field(FieldID *field_id, DexEngine &engine) {
        auto descriptor = ::get_dalvik_format(field_id->get_class()) + "->" + field_id->get_name_string() +
                          ::get_dalvik_format(field_id->get_type());
        auto external_field_provider = std::make_unique<DexExternalFieldProvider>(
                field_id->get_name(),
                descriptor,
                ::get_dalvik_format_string(field_id->get_class()),
                engine
        );
        auto external_field = std::make_unique<ExternalField>(*external_field_provider.get());

        this->dex_external_fields_providers.push_back(std::move(external_field_provider));
        this->ref_dex_externals_fields_providers.push_back(*this->dex_external_fields_providers.back().get());
        this->field_id_external_field_provider[field_id] = &(this->ref_dex_externals_fields_providers.back().get());

        this->sdk_external_fields.push_back(std::move(external_field));
        this->ref_sdk_externals_fields.push_back(*this->sdk_external_fields.back());
        this->field_id_external_field[field_id] = &(this->ref_sdk_externals_fields.back().get());
    }

    /**
     * @brief Get field by FieldID, creating external field if needed
     * @param field_id The field identifier to lookup
     * @param engine DexEngine reference for creating external fields
     * @return Field (if internal) or ExternalField (if external)
     *
     * Lookup order:
     * 1. Check internal fields first
     * 2. Check cached external fields
     * 3. Create external field if not found
     *
     * External fields represent fields from other DEX files.
     * They are created lazily and cached for performance.
     */
    field_external_field_t get_field_by_field_id(FieldID *field_id, DexEngine &engine) {
        auto it = field_id_field.find(field_id);

        if (it != field_id_field.end()) return *it->second;

        auto it_e = this->field_id_external_field.find(field_id);

        if (it_e != this->field_id_external_field.end()) return *it_e->second;

        save_external_field(field_id, engine);

        return *this->field_id_external_field[field_id];
    }
};

// ========================================
// DexEngine Implementation (Pimpl Bridge)
// ========================================

DexEngine::DexEngine(shuriken::io::ShurikenStream stream, Dex &owner_dex) : shuriken_stream(std::move(stream)),
                                                                            pimpl(std::make_unique<DexEngine::Impl>(
                                                                                    owner_dex)) {
    pimpl->disassembler = std::make_unique<InternalDisassembler>(this);
}

DexEngine::DexEngine(shuriken::io::ShurikenStream stream, std::string_view dex_path, Dex &owner_dex)
        : shuriken_stream(std::move(stream)),
          pimpl(std::make_unique<DexEngine::Impl>(owner_dex)) {
    this->pimpl->dex_path = dex_path;
    if (!dex_path.empty())
        this->pimpl->dex_name = std::filesystem::path(dex_path).filename().generic_string();
    pimpl->disassembler = std::make_unique<InternalDisassembler>(this);
}

DexEngine::~DexEngine() = default;

// ========================================
// Main Parsing Logic (converts DEX binary format to our object model)
// ========================================

shuriken::error::VoidResult DexEngine::parse() {
    auto &parser = this->pimpl->parser;

    // Parse the DEX file binary format
    auto result = parser.parse(shuriken_stream);
    if (!result) {
        return result;
    }

    // ========================================
    // Move parsed data from parser to engine storage
    // ========================================

    // fill the data with the information from the header
    pimpl->strings_pool = std::move(parser.get_strings_pool());
    pimpl->dex_type_providers = std::move(parser.get_types_pool());
    pimpl->sdk_dvmtypes = std::move(parser.get_dvm_types_pool());
    for (auto &dvm_type: pimpl->sdk_dvmtypes)
        pimpl->ref_sdk_dvmtypes.push_back(*dvm_type);
    pimpl->dex_prototypes_providers = std::move(parser.get_prototypes_pool());
    pimpl->sdk_prototypes = std::move(parser.get_dvm_prototype_pool());
    for (auto &sdk_prototype: pimpl->sdk_prototypes)
        pimpl->ref_sdk_prototypes.push_back(*sdk_prototype);

    // ========================================
    // Process each class definition from the DEX file
    // ========================================

    for (const auto &class_def: parser.get_classes()) {
        // Create the classes
        const DVMClass *class_id = ::as_class(*class_def->get_class_type());
        const DVMClass *parent_id = ::as_class(*class_def->get_superclass_type());
        std::vector<std::string> interfaces;
        for (const auto &interface: class_def->get_interfaces()) {
            const DVMClass *interface_type = ::as_class(*interface);
            interfaces.push_back(interface_type->get_dalvik_format_string());
        }
        auto [package, class_name] = split_class_descriptor(class_id->get_dalvik_format());

        auto new_class = std::make_unique<DexClassProvider>(
                class_name,
                package,
                class_id->get_dalvik_format(),
                class_id->get_canonical_name(),
                parent_id->get_dalvik_format(),
                interfaces
        );
        pimpl->save_class(new_class);

        auto &class_data_item = class_def->get_class_data_item();

        // ========================================
        // Generate the direct methods for this class
        // ========================================

        for (auto &encoded_method: class_data_item.get_direct_methods()) {
            MethodID &method_id = const_cast<MethodID &>(encoded_method.get_method_id());
            auto method_provider = std::make_unique<DexMethodProvider>(
                    method_id.get_name(),
                    encoded_method.get_access_flags(),
                    method_id.get_prototype(),
                    types::method_type_e::DIRECT_METHOD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this,
                    encoded_method.get_code_items()->get_registers_size(),
                    encoded_method.get_code_items()->get_bytecode(),
                    &encoded_method
            );
            pimpl->save_method(method_provider, &method_id);
        }

        // ========================================
        // Generate the virtual methods for this class
        // ========================================

        for (auto &encoded_method: class_data_item.get_virtual_methods()) {
            MethodID &method_id = const_cast<MethodID &>(encoded_method.get_method_id());
            auto method_provider = std::make_unique<DexMethodProvider>(
                    method_id.get_name(),
                    encoded_method.get_access_flags(),
                    method_id.get_prototype(),
                    types::method_type_e::VIRTUAL_METHOD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this,
                    encoded_method.get_code_items()->get_registers_size(),
                    encoded_method.get_code_items()->get_bytecode(),
                    &encoded_method
            );
            pimpl->save_method(method_provider, &method_id);
        }

        // ========================================
        // Generate the instance fields for this class
        // ========================================

        for (auto &encoded_field: class_data_item.get_instance_fields()) {
            FieldID &field_id = const_cast<FieldID &>(encoded_field.get_field());
            auto field_provider = std::make_unique<DexFieldProvider>(
                    field_id.get_name_string(),
                    field_id.get_type(),
                    encoded_field.get_flags(),
                    types::field_type_e::INSTANCE_FIELD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this
            );
            pimpl->save_field(field_provider, &field_id);
        }

        // ========================================
        // Generate the static fields for this class
        // ========================================

        for (auto &encoded_field: class_data_item.get_static_fields()) {
            FieldID &field_id = const_cast<FieldID &>(encoded_field.get_field());
            auto field_provider = std::make_unique<DexFieldProvider>(
                    field_id.get_name_string(),
                    field_id.get_type(),
                    encoded_field.get_flags(),
                    types::field_type_e::STATIC_FIELD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this
            );
            pimpl->save_field(field_provider, &field_id);
        }
    }

    for (auto & method_id : parser.get_methods_ids()) {
        pimpl->get_method_by_method_id(&method_id, *this);
    }

    for (auto & field_id : parser.get_fields_ids()) {
        pimpl->get_field_by_field_id(&field_id, *this);
    }

    return error::make_success();
}

// ========================================
// File Information Accessors
// ========================================

std::string_view shuriken::dex::DexEngine::get_dex_path() const {
    return this->pimpl->dex_path;
}

std::string shuriken::dex::DexEngine::get_dex_path_string() const {
    return this->pimpl->dex_path;
}

std::string_view shuriken::dex::DexEngine::get_dex_name() const {
    return this->pimpl->dex_name;
}

std::string shuriken::dex::DexEngine::get_dex_name_string() const {
    return this->pimpl->dex_name;
}

// ========================================
// String Pool Access
// ========================================

std::string_view shuriken::dex::DexEngine::get_string_by_id(size_t id) {
    if (id >= this->pimpl->strings_pool.size())
        return {};
    return this->pimpl->strings_pool[id];
}

size_t shuriken::dex::DexEngine::get_number_of_strings() const {
    return this->pimpl->strings_pool.size();
}

// ========================================
// Prototype Pool Access
// ========================================

DVMPrototype *shuriken::dex::DexEngine::get_prototype_by_id(size_t id) {
    if (id >= this->pimpl->ref_sdk_prototypes.size())
        return nullptr;
    return &this->pimpl->ref_sdk_prototypes[id].get();
}

size_t shuriken::dex::DexEngine::get_number_of_prototypes() const {
    return this->pimpl->ref_sdk_prototypes.size();
}

// ========================================
// Type Pool Access
// ========================================

DVMType *shuriken::dex::DexEngine::get_type_by_id(size_t id) {
    if (id >= this->pimpl->ref_sdk_dvmtypes.size())
        return nullptr;
    return &this->pimpl->ref_sdk_dvmtypes[id].get();
}

size_t shuriken::dex::DexEngine::get_number_of_types() const {
    return this->pimpl->ref_sdk_dvmtypes.size();
}

// ========================================
// Class Collection Access and Search
// ========================================

classes_deref_iterator_t shuriken::dex::DexEngine::get_classes() const {
    static classes_ref_t classes{this->pimpl->ref_sdk_classes};
    return classes;
}

size_t shuriken::dex::DexEngine::get_number_of_classes() const {
    return this->pimpl->ref_sdk_classes.size();
}

const Class *shuriken::dex::DexEngine::get_class_by_package_name_and_name(std::string_view package_name,
                                                                          std::string_view name) const {
    auto it = std::find_if(this->pimpl->sdk_classes.begin(),
                           this->pimpl->sdk_classes.end(),
                           [&](auto &c) -> bool {
                               // Dereference the unique_ptr first using the * operator
                               return c->get_package_name() == package_name && c->get_name() == name;
                           });

    if (it != this->pimpl->sdk_classes.end()) {
        // Return a pointer to the Class object inside the unique_ptr
        return it->get();
    }

    return nullptr;
}

Class *
shuriken::dex::DexEngine::get_class_by_package_name_and_name(std::string_view package_name, std::string_view name) {
    auto it = std::find_if(this->pimpl->sdk_classes.begin(),
                           this->pimpl->sdk_classes.end(),
                           [&](auto &c) -> bool {
                               // Dereference the unique_ptr first using the * operator
                               return c->get_package_name() == package_name && c->get_name() == name;
                           });

    if (it != this->pimpl->sdk_classes.end()) {
        // Return a pointer to the Class object inside the unique_ptr
        return it->get();
    }

    return nullptr;
}

const Class *shuriken::dex::DexEngine::get_class_by_descriptor(std::string_view descriptor) const {
    auto it = std::find_if(this->pimpl->sdk_classes.begin(),
                           this->pimpl->sdk_classes.end(),
                           [&](auto &c) -> bool {
                               return c->get_dalvik_name() == descriptor;
                           });

    if (it != this->pimpl->sdk_classes.end()) {
        // Return a pointer to the Class object inside the unique_ptr
        return it->get();
    }

    return nullptr;
}

Class *shuriken::dex::DexEngine::get_class_by_descriptor(std::string_view descriptor) {
    auto it = std::find_if(this->pimpl->sdk_classes.begin(),
                           this->pimpl->sdk_classes.end(),
                           [&](auto &c) -> bool {
                               return c->get_dalvik_name() == descriptor;
                           });

    if (it != this->pimpl->sdk_classes.end()) {
        // Return a pointer to the Class object inside the unique_ptr
        return it->get();
    }

    return nullptr;
}

std::vector<Class *> shuriken::dex::DexEngine::find_classes_by_regex(std::string_view descriptor_regex) {
    std::vector<Class *> matching_classes;
    std::regex pattern(descriptor_regex.data());

    for (const auto &cls: this->pimpl->sdk_classes) {
        std::string descriptor = cls->get_dalvik_name_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_classes.emplace_back(cls.get());
        }
    }

    return matching_classes;
}

// ========================================
// Method Collection Access and Search
// ========================================

method_deref_iterator_t shuriken::dex::DexEngine::get_methods() const {
    static methods_ref_t methods{this->pimpl->ref_sdk_methods};
    return methods;
}

external_methods_deref_iterator_t shuriken::dex::DexEngine::get_external_methods() const {
    static external_methods_ref_t external_methods{this->pimpl->ref_sdk_external_methods};
    return external_methods;
}

MethodID *shuriken::dex::DexEngine::get_method_by_id(size_t id) {
    if (id >= this->pimpl->parser.get_methods_ids().size())
        return nullptr;
    return &this->pimpl->parser.get_methods_ids()[id];
}

const MethodID *shuriken::dex::DexEngine::get_method_by_id(size_t id) const {
    if (id >= this->pimpl->parser.get_methods_ids().size())
        return nullptr;
    return &this->pimpl->parser.get_methods_ids()[id];
}

size_t shuriken::dex::DexEngine::get_number_of_methods() const {
    return this->pimpl->ref_sdk_methods.size();
}

const Method *
shuriken::dex::DexEngine::get_method_by_name_prototype(std::string_view name, std::string_view prototype) const {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto &m) {
        return m->get_name() == name && m->get_method_prototype().get_descriptor() == prototype;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

Method *shuriken::dex::DexEngine::get_method_by_name_prototype(std::string_view name, std::string_view prototype) {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto &m) {
        return m->get_name() == name && m->get_method_prototype().get_descriptor() == prototype;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

const Method *shuriken::dex::DexEngine::get_method_by_descriptor(std::string_view descriptor) const {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto &m) {
        return m->get_descriptor() == descriptor;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

Method *shuriken::dex::DexEngine::get_method_by_descriptor(std::string_view descriptor) {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto &m) {
        return m->get_descriptor() == descriptor;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

// ========================================
// Disassembly Service (converts bytecode to instruction objects)
// ========================================

void shuriken::dex::DexEngine::disassemble_method(DexMethodProvider &method) {
    auto instructions = this->pimpl->disassembler->disassemble(
            method.get_bytecode_vector());
    auto exceptions = this->pimpl->disassembler->determine_exception(method.get_encoded_method());
    method.set_method_instructions(instructions);
    method.set_exceptions(exceptions);
}

// ========================================
// Field Collection Access and Search
// ========================================

fields_deref_iterator_t shuriken::dex::DexEngine::get_fields() const {
    static fields_ref_t fields{this->pimpl->ref_sdk_fields};
    return fields;
}

external_fields_deref_iterator_t  shuriken::dex::DexEngine::get_external_fields() const {
    static external_fields_ref_t external_fields{this->pimpl->ref_sdk_externals_fields};
    return external_fields;
}

FieldID *shuriken::dex::DexEngine::get_field_by_id(size_t id) {
    if (id >= this->pimpl->parser.get_fields_ids().size())
        return nullptr;
    return &this->pimpl->parser.get_fields_ids()[id];
}

const FieldID *shuriken::dex::DexEngine::get_field_by_id(size_t id) const {
    if (id >= this->pimpl->parser.get_fields_ids().size())
        return nullptr;
    return &this->pimpl->parser.get_fields_ids()[id];
}

size_t shuriken::dex::DexEngine::get_number_of_fields() const {
    return this->pimpl->ref_sdk_fields.size();
}

const Field *shuriken::dex::DexEngine::get_field_by_name(std::string_view name) const {
    auto it = std::find_if(this->pimpl->sdk_fields.begin(), this->pimpl->sdk_fields.end(),
                           [&](const auto &field) -> bool {
                               return field->get_name() == name;
                           });

    if (it == this->pimpl->sdk_fields.end()) return nullptr;

    return it->get();
}

Field *shuriken::dex::DexEngine::get_field_by_name(std::string_view name) {
    auto it = std::find_if(this->pimpl->sdk_fields.begin(), this->pimpl->sdk_fields.end(),
                           [&](const auto &field) -> bool {
                               return field->get_name() == name;
                           });

    if (it == this->pimpl->sdk_fields.end()) return nullptr;

    return it->get();
}

// ========================================
// Regex Search Functions (for advanced filtering)
// ========================================

std::vector<Method *> shuriken::dex::DexEngine::found_method_by_regex(std::string_view descriptor_regex) {
    std::vector<Method *> matching_methods;
    std::regex pattern(descriptor_regex.data());

    for (const auto &method: this->pimpl->sdk_methods) {
        std::string descriptor = method->get_descriptor_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_methods.emplace_back(method.get());
        }
    }

    return matching_methods;
}

std::vector<Field *> shuriken::dex::DexEngine::found_field_by_regex(std::string_view descriptor_regex) {
    std::vector<Field *> matching_fields;
    std::regex pattern(descriptor_regex.data());

    for (const auto &field: this->pimpl->sdk_fields) {
        std::string descriptor = field->get_descriptor_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_fields.emplace_back(field.get());
        }
    }

    return matching_fields;
}