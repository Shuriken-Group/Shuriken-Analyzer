
#include "shuriken/sdk/dex/class.hpp"
#include "shuriken/internal/sdk/dex/class_impl.hpp"
#include <set>


using namespace shuriken::dex;

Class::Class(Class::Impl* impl) : impl(std::unique_ptr<Class::Impl>(impl)) {
}

std::string_view Class::get_name() const {
    return impl.get()->get_name();
}

std::string Class::get_name_string() const {
    return impl.get()->get_name_string();
}

std::string_view Class::get_package_name() const {
    return impl.get()->get_package_name();
}

std::string Class::get_package_name_string() const {
    return impl.get()->get_package_name_string();
}


std::string_view Class::get_dalvik_name() const {
    return impl.get()->get_dalvik_name();
}

std::string Class::get_dalvik_name_string() const {
    return impl.get()->get_dalvik_name_string();
}

std::string_view Class::get_canonical_name() const {
    return impl.get()->get_canonical_name();
}

std::string Class::get_canonical_name_string() const {
    return impl.get()->get_canonical_name_string();
}

std::string_view Class::get_extended_class() {
    return impl.get()->get_extended_class();
}

std::string Class::get_extended_class_string() {
    return impl.get()->get_extended_class_string();
}

std::size_t Class::get_number_of_implemented_classes() {
    return impl.get()->get_number_of_implemented_classes();
}

std::span<std::string> Class::get_implemented_classes() {
    return impl.get()->get_implemented_classes();
}

std::size_t Class::get_number_of_methods() const {
    return impl.get()->get_number_of_methods();
}

method_deref_iterator_t Class::get_methods() {
    return impl.get()->get_methods();
}

const Method *Class::get_method_by_name_prototype(std::string_view name, std::string_view prototype) const {
    return impl.get()->get_method_by_name_prototype(name, prototype);
}

Method *Class::get_method_by_name_prototype(std::string_view name, std::string_view prototype) {
    return impl.get()->get_method_by_name_prototype(name, prototype);
}

const Method *Class::get_method_by_descriptor(std::string_view descriptor) const {
    return impl.get()->get_method_by_descriptor(descriptor);
}

Method *Class::get_method_by_descriptor(std::string_view descriptor) {
    return impl.get()->get_method_by_descriptor(descriptor);
}

std::size_t Class::get_number_of_fields() const {
    return impl.get()->get_number_of_fields();
}

fields_deref_iterator_t Class::get_fields() {
    return impl.get()->get_fields();
}

const Field *Class::get_field_by_name(std::string_view name) const {
    return impl.get()->get_field_by_name(name);
}

Field *Class::get_field_by_name(std::string_view name) {
    return impl.get()->get_field_by_name(name);
}

std::vector<Method *> Class::found_method_by_regex(std::string_view descriptor_regex) {
    return impl.get()->found_method_by_regex(descriptor_regex);
}

std::vector<Field *> Class::found_field_by_regex(std::string_view descriptor_regex) {
    return impl.get()->found_field_by_regex(descriptor_regex);
}

shuriken::iterator_range<span_method_idx_iterator_t> Class::get_xref_new_instance_methods() {
    span_method_idx_t empty{};
    return shuriken::iterator_range<span_method_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_method_idx_iterator_t> Class::get_xref_const_class() {
    span_method_idx_t empty{};
    return shuriken::iterator_range<span_method_idx_iterator_t>(empty);
}

