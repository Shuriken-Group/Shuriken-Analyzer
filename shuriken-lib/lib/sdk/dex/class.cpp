
#include <shuriken/sdk/dex/class.h>
#include <unordered_map>
#include <set>

using namespace shuriken::dex;

Class::Class(DexClassProvider &dex_class_provider) : dex_class_provider(dex_class_provider){
}

std::string_view Class::get_name() const {
    return std::string_view();
}

std::string Class::get_name_string() const {
    return std::string();
}

class_external_class_t Class::get_extended_class() {
    return std::monostate{};
}


std::size_t Class::get_number_of_implemented_classes() {
    return 0;
}

shuriken::iterator_range<span_class_external_class_iterator_t> Class::get_implemented_classes() {
    span_class_external_class_t empty{};
    return shuriken::iterator_range<span_class_external_class_iterator_t>(empty);
}

std::size_t Class::get_number_of_methods() const {
    return 0;
}

shuriken::iterator_range<methods_ref_iterator_t> Class::get_methods() {
    methods_ref_t empty{};
    return shuriken::iterator_range<methods_ref_iterator_t>(empty);
}

const Method *Class::get_method_by_name_prototype(const std::string &name, const std::string &prototype) const {
    return nullptr;
}

Method *Class::get_method_by_name_prototype(const std::string &name, const std::string &prototype) {
    return nullptr;
}

const Method *Class::get_method_by_descriptor(const std::string &descriptor) const {
    return nullptr;
}

Method *Class::get_method_by_descriptor(const std::string &descriptor) {
    return nullptr;
}

std::size_t Class::get_number_of_fields() const {
    return 0;
}

shuriken::iterator_range<fields_ref_iterator_t> Class::get_fields() {
    fields_ref_t empty{};
    return shuriken::iterator_range<fields_ref_iterator_t>(empty);
}

const Field *Class::get_field_by_name(const std::string &name) const {
    return nullptr;
}

Field *Class::get_field_by_name(const std::string &name) {
    return nullptr;
}

std::vector<Method *> Class::found_method_by_regex(const std::string &descriptor_regex) {
    return std::vector<Method *>();
}

std::vector<Field *> Class::found_field_by_regex(const std::string &descriptor_regex) {
    return std::vector<Field *>();
}

shuriken::iterator_range<span_method_idx_iterator_t> Class::get_xref_new_instance_methods() {
    span_method_idx_t empty{};
    return shuriken::iterator_range<span_method_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_method_idx_iterator_t> Class::get_xref_const_class() {
    span_method_idx_t empty{};
    return shuriken::iterator_range<span_method_idx_iterator_t>(empty);
}

