//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <memory>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <regex>

#include <shuriken/internal/engine/dex/dex_engine.hpp>

#include "shuriken/sdk/dex/dex.hpp"
#include "shuriken/sdk/dex/class.hpp"
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/sdk/dex/field.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/internal/providers/dex/dex_class_provider.hpp"
#include "shuriken/internal/providers/dex/dex_method_provider.hpp"
#include "shuriken/internal/providers/dex/dex_field_provider.hpp"
#include "shuriken/internal/providers/dex/dvm_types_provider.hpp"
#include "shuriken/internal/providers/dex/dvm_prototypes_provider.hpp"
#include "shuriken/internal/providers/dex/custom_types.hpp"

#include "shuriken/internal/engine/dex/parser/parser.hpp"


using namespace shuriken::dex;


namespace {
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

class DexEngine::Impl {
public:
    std::reference_wrapper<Dex> owner_dex;

    std::string dex_path;
    std::string dex_name;
    // ownership of classes
    std::vector<std::unique_ptr<Class>> sdk_classes;
    std::vector<std::unique_ptr<DexClassProvider>> dex_class_providers;

    // ownership of methods
    std::vector<std::unique_ptr<Method>> sdk_methods;
    std::vector<std::unique_ptr<DexMethodProvider>> dex_methods_providers;

    // ownership of fields
    std::vector<std::unique_ptr<Field>> sdk_fields;
    std::vector<std::unique_ptr<DexFieldProvider>> dex_fields_providers;

    // ownership of prototypes
    std::vector<std::unique_ptr<DVMPrototype>> sdk_prototypes;
    std::vector<std::unique_ptr<DVMPrototypeProvider>> dex_prototypes_providers;

    // ownership of types
    std::vector<std::unique_ptr<DVMType>> sdk_dvmtypes;
    std::vector<std::unique_ptr<DVMTypeProvider>> dex_type_providers;

    // cache of all the previous vectors
    std::vector<std::reference_wrapper<Class>> ref_sdk_classes;
    std::vector<std::reference_wrapper<DexClassProvider>> ref_dex_class_providers;
    std::vector<std::reference_wrapper<Method>> ref_sdk_methods;
    std::vector<std::reference_wrapper<DexMethodProvider>> ref_dex_methods_providers;
    std::vector<std::reference_wrapper<Field>> ref_sdk_fields;
    std::vector<std::reference_wrapper<DexFieldProvider>> ref_dex_fields_providers;
    std::vector<std::reference_wrapper<DVMPrototype>> ref_sdk_prototypes;
    std::vector<std::reference_wrapper<DVMPrototypeProvider>> ref_dex_prototypes_providers;
    std::vector<std::reference_wrapper<DVMType>> ref_sdk_dvmtypes;
    std::vector<std::reference_wrapper<DVMTypeProvider>> ref_dex_type_providers;

    Impl(Dex &owner_dex) : owner_dex(owner_dex) {}
    ~Impl() = default;
};

DexEngine::DexEngine(shuriken::io::ShurikenStream stream, Dex &owner_dex) : shuriken_stream(std::move(stream)),
                                                                            pimpl(std::make_unique<DexEngine::Impl>(owner_dex)) {
}

DexEngine::DexEngine(shuriken::io::ShurikenStream stream, std::string_view dex_path, Dex &owner_dex)
        : shuriken_stream(std::move(stream)),
          pimpl(std::make_unique<DexEngine::Impl>(owner_dex)) {
    this->pimpl->dex_path = dex_path;
    if (!dex_path.empty())
        this->pimpl->dex_name = std::filesystem::path(dex_path).filename();
}

DexEngine::~DexEngine() = default;

shuriken::error::VoidResult DexEngine::parse() {
    Parser parser;
    auto result = parser.parse(shuriken_stream);
    if (!result) {
        return result;
    }

    // fill the data with the information from the header
    pimpl->dex_type_providers = std::move(parser.get_types_pool());
    pimpl->sdk_dvmtypes = std::move(parser.get_dvm_types_pool());
    pimpl->dex_prototypes_providers = std::move(parser.get_prototypes_pool());
    pimpl->sdk_prototypes = std::move(parser.get_dvm_prototype_pool());

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
        auto new_sdk_class = std::make_unique<Class>(*new_class);
        pimpl->dex_class_providers.push_back(std::move(new_class));
        pimpl->ref_dex_class_providers.push_back(std::ref(*pimpl->dex_class_providers.back().get()));
        pimpl->sdk_classes.push_back(std::move(new_sdk_class));
        pimpl->ref_sdk_classes.push_back(std::ref(*pimpl->sdk_classes.back().get()));

        auto & class_data_item = class_def->get_class_data_item();

        // generate the methods
        for (auto & encoded_method : class_data_item.get_direct_methods()) {
            MethodID & method_id = const_cast<MethodID&>(encoded_method.get_method_id());
            auto method_provider = std::make_unique<DexMethodProvider>(
                        method_id.get_name(),
                        encoded_method.get_access_flags(),
                        method_id.get_prototype(),
                        types::method_type_e::DIRECT_METHOD,
                        pimpl->ref_sdk_classes.back(),
                        pimpl->owner_dex,
                        *this
                    );
            auto method_sdk = std::make_unique<Method>(*method_provider.get());
            pimpl->dex_methods_providers.push_back(std::move(method_provider));
            pimpl->ref_dex_methods_providers.push_back(*pimpl->dex_methods_providers.back());
            pimpl->sdk_methods.push_back(std::move(method_sdk));
            pimpl->ref_sdk_methods.push_back(*pimpl->sdk_methods.back());
            pimpl->dex_class_providers.back()->add_method(pimpl->ref_sdk_methods.back());
        }

        for (auto & encoded_method : class_data_item.get_virtual_methods()) {
            MethodID & method_id = const_cast<MethodID&>(encoded_method.get_method_id());
            auto method_provider = std::make_unique<DexMethodProvider>(
                    method_id.get_name(),
                    encoded_method.get_access_flags(),
                    method_id.get_prototype(),
                    types::method_type_e::VIRTUAL_METHOD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this
            );
            auto method_sdk = std::make_unique<Method>(*method_provider.get());
            pimpl->dex_methods_providers.push_back(std::move(method_provider));
            pimpl->ref_dex_methods_providers.push_back(*pimpl->dex_methods_providers.back());
            pimpl->sdk_methods.push_back(std::move(method_sdk));
            pimpl->ref_sdk_methods.push_back(*pimpl->sdk_methods.back());
            pimpl->dex_class_providers.back()->add_method(pimpl->ref_sdk_methods.back());
        }

        for (auto & encoded_field : class_data_item.get_instance_fields()) {
            FieldID & field_id = const_cast<FieldID&>(encoded_field.get_field());
            auto field_provider = std::make_unique<DexFieldProvider>(
                    field_id.get_name_string(),
                    field_id.get_type(),
                    encoded_field.get_flags(),
                    types::field_type_e::INSTANCE_FIELD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this
                    );
            auto field_sdk = std::make_unique<Field>(*field_provider.get());
            pimpl->dex_fields_providers.push_back(std::move(field_provider));
            pimpl->ref_dex_fields_providers.push_back(*pimpl->dex_fields_providers.back());
            pimpl->sdk_fields.push_back(std::move(field_sdk));
            pimpl->ref_sdk_fields.push_back(*pimpl->sdk_fields.back());
            pimpl->dex_class_providers.back()->add_field(pimpl->ref_sdk_fields.back());
        }

        for (auto & encoded_field : class_data_item.get_static_fields()) {
            FieldID & field_id = const_cast<FieldID&>(encoded_field.get_field());
            auto field_provider = std::make_unique<DexFieldProvider>(
                    field_id.get_name_string(),
                    field_id.get_type(),
                    encoded_field.get_flags(),
                    types::field_type_e::STATIC_FIELD,
                    pimpl->ref_sdk_classes.back(),
                    pimpl->owner_dex,
                    *this
            );
            auto field_sdk = std::make_unique<Field>(*field_provider.get());
            pimpl->dex_fields_providers.push_back(std::move(field_provider));
            pimpl->ref_dex_fields_providers.push_back(*pimpl->dex_fields_providers.back());
            pimpl->sdk_fields.push_back(std::move(field_sdk));
            pimpl->ref_sdk_fields.push_back(*pimpl->sdk_fields.back());
            pimpl->dex_class_providers.back()->add_field(pimpl->ref_sdk_fields.back());
        }
    }


    return error::make_success();
}

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

classes_deref_iterator_t shuriken::dex::DexEngine::get_classes() const {
    static classes_ref_t classes{this->pimpl->ref_sdk_classes};
    return classes;
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

method_deref_iterator_t shuriken::dex::DexEngine::get_methods() const {
    static methods_ref_t methods{this->pimpl->ref_sdk_methods};
    return methods;
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

fields_deref_iterator_t shuriken::dex::DexEngine::get_fields() const {
    static fields_ref_t fields{this->pimpl->ref_sdk_fields};
    return fields;
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






