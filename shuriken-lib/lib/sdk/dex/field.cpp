
#include "shuriken/sdk/dex/field.hpp"

using namespace shuriken::dex;

Field::Field(DexFieldProvider &dex_field_provider) : dex_field_provider(dex_field_provider){
}

std::string_view Field::get_name() const {
    return std::string();
}

std::string Field::get_name_string() const {
    return std::string();
}

types::access_flags Field::get_field_access_flags() const {
    return types::ACC_ENUM;
}

const DVMType *Field::get_field_type() const {
    return nullptr;
}

DVMType *Field::get_field_type() {
    return nullptr;
}

const Class *Field::get_owner_class() const {
    return nullptr;
}

Class *Field::get_owner_class() {
    return nullptr;
}

const Dex *Field::get_owner_dex() const {
    return nullptr;
}

Dex *Field::get_owner_dex() {
    return nullptr;
}

std::string_view Field::get_descriptor() const {
    return std::string_view();
}

std::string Field::get_descriptor_string() const {
    return std::string();
}

shuriken::iterator_range<span_class_method_idx_iterator_t> Field::get_xref_read() {
    span_class_method_idx_t empty{};
    return make_range(empty.begin(), empty.end());
}

shuriken::iterator_range<span_class_method_idx_iterator_t> Field::get_xref_write() {
    span_class_method_idx_t empty{};
    return make_range(empty.begin(), empty.end());
}


