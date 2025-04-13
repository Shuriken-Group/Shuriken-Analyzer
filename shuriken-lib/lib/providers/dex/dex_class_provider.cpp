//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <algorithm>
#include <ranges>
#include <regex>

#include "shuriken/internal/providers/dex/dex_class_provider.hpp"
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/sdk/dex/field.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"

using namespace shuriken::dex;

namespace {
    std::string dalvik_to_canonical(const std::string &dalvik_format) {
        // Skip the leading 'L' and remove trailing ';'
        std::string result = dalvik_format.substr(1, dalvik_format.length() - 2);

        // Replace all '/' with '.'
        size_t pos = 0;
        while ((pos = result.find('/', pos)) != std::string::npos) {
            result.replace(pos, 1, ".");
            pos += 1; // Move past the replacement
        }

        return result;
    }
}

DexClassProvider::DexClassProvider(std::string_view name, std::string_view package_name, std::string_view dalvik_format,
                                   std::string_view canonical_name,
                                   std::string_view extended_class, std::vector<std::string> &implemented_classes) :
        name(name), package_name(package_name), dalvik_format(dalvik_format), canonical_format(canonical_name),
        extended_class(extended_class),
        implemented_classes(std::move(implemented_classes)) {
}

std::string_view DexClassProvider::get_name() const {
    return name;
}

std::string DexClassProvider::get_name_string() const {
    return name;
}

std::string_view DexClassProvider::get_package_name() const {
    return package_name;
}

std::string DexClassProvider::get_package_name_string() const {
    return package_name;
}

std::string_view DexClassProvider::get_dalvik_name() const {
    return dalvik_format;
}

std::string DexClassProvider::get_dalvik_name_string() const {
    return dalvik_format;
}

std::string_view DexClassProvider::get_canonical_name() const {
    return canonical_format;
}

std::string DexClassProvider::get_canonical_name_string() const {
    return canonical_format;
}

std::string_view DexClassProvider::get_extended_class() {
    return extended_class;
}

std::string DexClassProvider::get_extended_class_string() {
    return extended_class;
}

std::size_t DexClassProvider::get_number_of_implemented_classes() {
    return implemented_classes.size();
}

std::span<std::string> DexClassProvider::get_implemented_classes() {
    static std::span<std::string> implemented{implemented_classes};
    return implemented;
}

void DexClassProvider::add_method(method_t method) {
    methods.emplace_back(method);
}

void DexClassProvider::add_field(field_t field) {
    fields.emplace_back(field);
}

std::size_t DexClassProvider::get_number_of_methods() const {
    return methods.size();
}

method_deref_iterator_t DexClassProvider::get_methods() {
    static std::span<method_t> m{methods};
    return deref_iterator_range{m};
}

const Method *
DexClassProvider::get_method_by_name_prototype(std::string_view name, std::string_view prototype) const {
    auto it = std::find_if(methods.begin(), methods.end(), [&](const method_t &m) {
        return m.get().get_name() == name && m.get().get_method_prototype().get_descriptor() == prototype;
    });

    if (it == methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return &(it->get());
}

Method *DexClassProvider::get_method_by_name_prototype(std::string_view name, std::string_view prototype) {
    auto it = std::find_if(methods.begin(), methods.end(), [&](const method_t &m) {
        return m.get().get_name() == name && m.get().get_method_prototype().get_descriptor() == prototype;
    });

    if (it == methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return &(it->get());
}

const Method *DexClassProvider::get_method_by_descriptor(std::string_view descriptor) const {
    auto it = std::find_if(methods.begin(), methods.end(), [&](const method_t &m) {
        return m.get().get_descriptor() == descriptor;
    });

    if (it == methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return &(it->get());
}

Method *DexClassProvider::get_method_by_descriptor(std::string_view descriptor) {
    auto it = std::find_if(methods.begin(), methods.end(), [&](const method_t &m) {
        return m.get().get_descriptor() == descriptor;
    });

    if (it == methods.end()) return nullptr;

    // Assuming method_t::get() returns a Method or a reference to Method
    return &(it->get());
}

std::size_t DexClassProvider::get_number_of_fields() const {
    return fields.size();
}

fields_deref_iterator_t DexClassProvider::get_fields() {
    static std::span<field_t> f{fields};
    return f;
}

const Field *DexClassProvider::get_field_by_name(std::string_view name) const {
    auto it = std::find_if(fields.begin(), fields.end(), [&](const field_t &f) {
        return f.get().get_name() == name;
    });

    if (it == fields.end()) return nullptr;

    return &(it->get());
}

Field *DexClassProvider::get_field_by_name(std::string_view name) {
    auto it = std::find_if(fields.begin(), fields.end(), [&](const field_t &f) {
        return f.get().get_name() == name;
    });

    if (it == fields.end()) return nullptr;

    return &(it->get());
}

std::vector<Method *> DexClassProvider::found_method_by_regex(std::string_view descriptor_regex) {
    std::vector<Method *> matching_methods;
    std::regex pattern(descriptor_regex.data());

    for (const auto &method: methods) {
        std::string descriptor = method.get().get_descriptor_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_methods.emplace_back(&(method.get()));
        }
    }

    return matching_methods;
}

std::vector<Field *> DexClassProvider::found_field_by_regex(std::string_view descriptor_regex) {
    std::vector<Field *> matching_fields;
    std::regex pattern(descriptor_regex.data());

    for (const auto &field: fields) {
        std::string descriptor = field.get().get_descriptor_string();
        if (std::regex_match(descriptor, pattern)) {
            matching_fields.emplace_back(&(field.get()));
        }
    }

    return matching_fields;
}


