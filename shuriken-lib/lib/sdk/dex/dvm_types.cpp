
#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/internal/providers/dex/dvm_types_provider.hpp>

using namespace shuriken::dex;

DVMType::DVMType(DVMTypeProvider &dvm_type_provider) : dvm_type_provider(dvm_type_provider) {
}


DVMFundamental::DVMFundamental(DVMFundamentalProvider &dvm_fundamental_provider) : DVMType(dvm_fundamental_provider),
                                                                                   dvm_fundamental_provider(
                                                                                           dvm_fundamental_provider) {
}

types::type_e DVMFundamental::get_type() const {
    return types::type_e::FUNDAMENTAL;
}

std::string_view DVMFundamental::get_dalvik_format() const {
    return std::string_view();
}

std::string DVMFundamental::get_dalvik_format_string() const {
    return std::string();
}

std::string_view DVMFundamental::get_canonical_name() const {
    return std::string_view();
}

std::string DVMFundamental::get_canonical_name_string() const {
    return std::string();
}

types::fundamental_e DVMFundamental::get_fundamental_type() const {
    return types::fundamental_e::LONG;
}


DVMClass::DVMClass(DVMClassProvider &dvm_class_provider) : DVMType(dvm_class_provider),
                                                           dvm_class_provider(dvm_class_provider) {
}

types::type_e DVMClass::get_type() const {
    return types::type_e::CLASS;
}

std::string_view DVMClass::get_dalvik_format() const {
    return std::string_view();
}

std::string DVMClass::get_dalvik_format_string() const {
    return std::string();
}

std::string_view DVMClass::get_canonical_name() const {
    return std::string_view();
}

std::string DVMClass::get_canonical_name_string() const {
    return std::string();
}

DVMArray::DVMArray(DVMArrayProvider &dvm_array_provider) : DVMType(dvm_array_provider), dvm_array_provider(dvm_array_provider) {
}

types::type_e DVMArray::get_type() const {
    return types::type_e::ARRAY;
}

std::string_view DVMArray::get_dalvik_format() const {
    return std::string_view();
}

std::string DVMArray::get_dalvik_format_string() const {
    return std::string();
}

std::string_view DVMArray::get_canonical_name() const {
    return std::string_view();
}

std::string DVMArray::get_canonical_name_string() const {
    return std::string();
}

size_t DVMArray::get_array_depth() const {
    return 0;
}

const DVMType *DVMArray::get_base_type() const {
    return nullptr;
}

