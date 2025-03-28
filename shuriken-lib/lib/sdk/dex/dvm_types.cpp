
#include <variant>  // For std::visit and std::variant
#include <utility>  // For std::forward if you need it

#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/internal/providers/dex/dvm_types_provider.hpp>

using namespace shuriken::dex;


DVMFundamental::DVMFundamental(DVMFundamentalProvider &dvm_fundamental_provider) :
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


DVMClass::DVMClass(DVMClassProvider &dvm_class_provider) :
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

DVMArray::DVMArray(DVMArrayProvider &dvm_array_provider) : dvm_array_provider(dvm_array_provider) {
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

// Helper template for std::visit
template<class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

// Deduction guide (C++17 or later)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

types::type_e get_type(const DVMType& type) {
    return std::visit(overloaded {
            [](const DVMFundamental&) { return types::type_e::FUNDAMENTAL; },
            [](const DVMClass&) { return types::type_e::CLASS; },
            [](const DVMArray&) { return types::type_e::ARRAY; },
    }, type);
}

std::string_view get_dalvik_format_string(const DVMType& type) {
    return std::visit(overloaded {
            [](const DVMFundamental& t) { return t.get_dalvik_format(); },
            [](const DVMClass& t) { return t.get_dalvik_format(); },
            [](const DVMArray& t) { return t.get_dalvik_format(); },
    }, type);
}

std::string get_dalvik_format(const DVMType& type) {
    return std::visit(overloaded {
            [](const DVMFundamental& t) { return t.get_dalvik_format_string(); },
            [](const DVMClass& t) { return t.get_dalvik_format_string(); },
            [](const DVMArray& t) { return t.get_dalvik_format_string(); },
    }, type);
}

std::string_view get_canonical_name(const DVMType& type) {
    return std::visit(overloaded {
            [](const DVMFundamental& t) { return t.get_canonical_name(); },
            [](const DVMClass& t) { return t.get_canonical_name(); },
            [](const DVMArray& t) { return t.get_canonical_name(); },
    }, type);
}

std::string get_canonical_name_string(const DVMType& type) {
    return std::visit(overloaded {
            [](const DVMFundamental& t) { return t.get_canonical_name_string(); },
            [](const DVMClass& t) { return t.get_canonical_name_string(); },
            [](const DVMArray& t) { return t.get_canonical_name_string(); },
    }, type);
}

const DVMFundamental * as_fundamental(const DVMType& type) {
    return std::get_if<DVMFundamental>(&type);
}

const DVMClass * as_class(const DVMType& type) {
    return std::get_if<DVMClass>(&type);
}

const DVMArray * as_array(const DVMType& type) {
    return std::get_if<DVMArray>(&type);
}

