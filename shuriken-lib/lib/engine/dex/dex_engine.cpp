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


using namespace shuriken::dex;

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

    Impl(Dex& owner_dex) : owner_dex(owner_dex) {}
    ~Impl() = default;
};

DexEngine::DexEngine(shuriken::io::ShurikenStream stream, Dex& owner_dex) : shuriken_stream(std::move(stream)),
            pimpl(new DexEngine::Impl(owner_dex)) {
}

shuriken::dex::DexEngine::DexEngine(shuriken::io::ShurikenStream stream, std::string_view dex_path, Dex& owner_dex) : shuriken_stream(std::move(stream)),
                                                                                                      pimpl(new DexEngine::Impl(owner_dex)) {
    this->pimpl->dex_path = dex_path;
    if (!dex_path.empty())
        this->pimpl->dex_name = std::filesystem::path(dex_path).filename();
}

shuriken::dex::DexEngine::~DexEngine() {
    delete pimpl;
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
                           [&](auto & c)-> bool {
                       // Dereference the unique_ptr first using the * operator
                       return c->get_package_name() == package_name && c->get_name() == name;
    });

    if (it != this->pimpl->sdk_classes.end()) {
        // Return a pointer to the Class object inside the unique_ptr
        return it->get();
    }

    return nullptr;
}

Class *shuriken::dex::DexEngine::get_class_by_package_name_and_name(std::string_view package_name, std::string_view name) {
    auto it = std::find_if(this->pimpl->sdk_classes.begin(),
                           this->pimpl->sdk_classes.end(),
                           [&](auto & c)-> bool {
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
                           [&](auto & c) -> bool {
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
                           [&](auto & c) -> bool {
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

    for (const auto & cls : this->pimpl->sdk_classes) {
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
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto& m) {
        return m->get_name() == name && m->get_method_prototype().get_descriptor() == prototype;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

Method *shuriken::dex::DexEngine::get_method_by_name_prototype(std::string_view name, std::string_view prototype) {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto& m) {
        return m->get_name() == name && m->get_method_prototype().get_descriptor() == prototype;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

const Method *shuriken::dex::DexEngine::get_method_by_descriptor(std::string_view descriptor) const {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto& m) {
        return m->get_descriptor() == descriptor;
    });

    if (it == this->pimpl->sdk_methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return it->get();
}

Method *shuriken::dex::DexEngine::get_method_by_descriptor(std::string_view descriptor) {
    auto it = std::find_if(this->pimpl->sdk_methods.begin(), this->pimpl->sdk_methods.end(), [&](const auto& m) {
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
    auto it = std::find_if(this->pimpl->sdk_fields.begin(), this->pimpl->sdk_fields.end(), [&](const auto& field) -> bool{
        return field->get_name() == name;
    });

    if (it == this->pimpl->sdk_fields.end()) return nullptr;

    return it->get();
}

Field *shuriken::dex::DexEngine::get_field_by_name(std::string_view name) {
    auto it = std::find_if(this->pimpl->sdk_fields.begin(), this->pimpl->sdk_fields.end(), [&](const auto& field) -> bool{
        return field->get_name() == name;
    });

    if (it == this->pimpl->sdk_fields.end()) return nullptr;

    return it->get();
}

std::vector<Method *> shuriken::dex::DexEngine::found_method_by_regex(std::string_view descriptor_regex) {
    std::vector<Method*> matching_methods;
    std::regex pattern(descriptor_regex.data());

    for (const auto& method : this->pimpl->sdk_methods) {
        std::string descriptor = method->get_descriptor_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_methods.emplace_back(method.get());
        }
    }

    return matching_methods;
}

std::vector<Field *> shuriken::dex::DexEngine::found_field_by_regex(std::string_view descriptor_regex) {
    std::vector<Field*> matching_fields;
    std::regex pattern(descriptor_regex.data());

    for (const auto& field : this->pimpl->sdk_fields) {
        std::string descriptor = field->get_descriptor_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_fields.emplace_back(field.get());
        }
    }

    return matching_fields;
}






