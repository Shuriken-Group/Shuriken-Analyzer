
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/internal/providers/dex/dex_method_provider.hpp"

using namespace shuriken::dex;

Method::Method(DexMethodProvider & dex_method_provider) : dex_method_provider(dex_method_provider){
}

std::string_view Method::get_name() const {
    return dex_method_provider.get().get_name();
}

std::string Method::get_name_string() const {
    return  dex_method_provider.get().get_name_string();
}

types::access_flags Method::get_method_access_flags() const {
    return dex_method_provider.get().get_method_access_flags();
}

std::string_view Method::get_method_access_flags_str() {
    return dex_method_provider.get().get_method_access_flags_str();
}

const DVMPrototype &Method::get_method_prototype() const {
    return dex_method_provider.get().get_method_prototype();
}

DVMPrototype &Method::get_method_prototype() {
    return dex_method_provider.get().get_method_prototype();
}

types::method_type_e Method::get_method_type() const {
    return dex_method_provider.get().get_method_type();
}

const Class &Method::get_owner_class() const {
    return dex_method_provider.get().get_owner_class();
}

Class &Method::get_owner_class() {
    return dex_method_provider.get().get_owner_class();
}

const Dex &Method::get_owner_dex() const {
    return dex_method_provider.get().get_owner_dex();
}

Dex &Method::get_owner_dex() {
    return dex_method_provider.get().get_owner_dex();
}

std::string_view Method::get_descriptor() const {
    return dex_method_provider.get().get_descriptor();
}

std::string Method::get_descriptor_string() const {
    return dex_method_provider.get().get_descriptor_string();
}

std::uint16_t Method::registers_size() const {
    return dex_method_provider.get().registers_size();
}

std::span<const std::uint8_t> Method::get_bytecode() const {
    return dex_method_provider.get().get_bytecode();
}

std::list<std::reference_wrapper<Instruction>> & Method::get_method_instructions() {
    return dex_method_provider.get().get_method_instructions();
}

disassembler::exceptions_data_t & Method::get_exceptions() {
    return dex_method_provider.get().get_exceptions();
}

shuriken::iterator_range<span_class_field_idx_iterator_t> Method::get_xref_read_fields_in_method() {
    span_class_field_idx_t empty{};
    return shuriken::iterator_range<span_class_field_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_class_field_idx_iterator_t> Method::get_xref_written_fields_in_method() {
    span_class_field_idx_t empty{};
    return shuriken::iterator_range<span_class_field_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_class_method_idx_iterator_t> Method::get_xref_methods_called() {
    span_class_method_idx_t empty{};
    return shuriken::iterator_range<span_class_method_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_class_method_idx_iterator_t> Method::get_xref_caller_methods() {
    span_class_method_idx_t empty{};
    return shuriken::iterator_range<span_class_method_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_class_idx_iterator_t> Method::get_xref_new_instance_classes() {
    span_class_idx_t empty{};
    return shuriken::iterator_range<span_class_idx_iterator_t>(empty);
}

shuriken::iterator_range<span_class_idx_iterator_t> Method::get_xref_const_class() {
    span_class_idx_t empty{};
    return shuriken::iterator_range<span_class_idx_iterator_t>(empty);
}

