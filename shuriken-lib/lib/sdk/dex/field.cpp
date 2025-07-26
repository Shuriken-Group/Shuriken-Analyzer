#include "shuriken/sdk/dex/field.hpp"
#include "shuriken/internal/sdk/dex/field_impl.hpp"

#include <memory>

using namespace shuriken::dex;

Field::Field(Field::Impl *impl) : impl(std::unique_ptr<Field::Impl>(impl)){
}

std::string_view Field::get_name() const {
    return impl.get()->get_name();
}

std::string Field::get_name_string() const {
    return impl.get()->get_name_string();
}

types::access_flags Field::get_field_access_flags() const {
    return impl.get()->get_field_access_flags();
}

std::string_view Field::get_field_access_flags_str() const {
    return impl.get()->get_field_access_flags_str();
}

types::field_type_e Field::get_type() const {
    return impl.get()->get_type();
}

const DVMType & Field::get_field_type() const {
    return impl.get()->get_field_type();
}

DVMType & Field::get_field_type() {
    return impl.get()->get_field_type();
}

const Class &Field::get_owner_class() const {
    return impl.get()->get_owner_class();
}

Class &Field::get_owner_class() {
    return impl.get()->get_owner_class();
}

const Dex &Field::get_owner_dex() const {
    return impl.get()->get_owner_dex();
}

Dex &Field::get_owner_dex() {
    return impl.get()->get_owner_dex();
}

std::string_view Field::get_descriptor() const {
    return impl.get()->get_descriptor();
}

std::string Field::get_descriptor_string() const {
    return impl.get()->get_descriptor_string();
}

shuriken::iterator_range<span_class_method_idx_iterator_t> Field::get_xref_read() {
    span_class_method_idx_t empty{};
    return make_range(empty.begin(), empty.end());
}

shuriken::iterator_range<span_class_method_idx_iterator_t> Field::get_xref_write() {
    span_class_method_idx_t empty{};
    return make_range(empty.begin(), empty.end());
}


