
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/internal/sdk/dex/method_impl.hpp"
#include "shuriken/internal/sdk/dex/instruction_impl.hpp"

#include <memory>

using namespace shuriken::dex;

Method::Method(Impl * impl) : impl(std::unique_ptr<Impl>(impl)){
}

std::string_view Method::get_name() const {
    return impl.get()->get_name();
}

std::string Method::get_name_string() const {
    return  impl.get()->get_name_string();
}

types::access_flags Method::get_method_access_flags() const {
    return impl.get()->get_method_access_flags();
}

std::string_view Method::get_method_access_flags_str() {
    return impl.get()->get_method_access_flags_str();
}

const DVMPrototype &Method::get_method_prototype() const {
    return impl.get()->get_method_prototype();
}

DVMPrototype &Method::get_method_prototype() {
    return impl.get()->get_method_prototype();
}

types::method_type_e Method::get_method_type() const {
    return impl.get()->get_method_type();
}

const Class &Method::get_owner_class() const {
    return impl.get()->get_owner_class();
}

Class &Method::get_owner_class() {
    return impl.get()->get_owner_class();
}

const Dex &Method::get_owner_dex() const {
    return impl.get()->get_owner_dex();
}

Dex &Method::get_owner_dex() {
    return impl.get()->get_owner_dex();
}

std::string_view Method::get_descriptor() const {
    return impl.get()->get_descriptor();
}

std::string Method::get_descriptor_string() const {
    return impl.get()->get_descriptor_string();
}

std::uint16_t Method::registers_size() const {
    return impl.get()->registers_size();
}

std::span<std::uint8_t> Method::get_bytecode() {
    return impl.get()->get_bytecode();
}

std::list<std::reference_wrapper<Instruction>> & Method::get_method_instructions() {
    return impl.get()->get_method_instructions();
}

disassembler::exceptions_data_t & Method::get_exceptions() {
    return impl.get()->get_exceptions();
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

