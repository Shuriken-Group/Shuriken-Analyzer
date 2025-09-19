
#include "shuriken/internal/sdk/dex/external_class_impl.hpp"

using namespace shuriken::dex;

ExternalClass::ExternalClass(Impl *i) : impl(std::unique_ptr<Impl>(i)) {
}

std::string_view ExternalClass::get_name() const {
    return impl->get_name();
}

std::string ExternalClass::get_name_string() const {
    return impl->get_name_string();
}

std::string_view ExternalClass::get_package_name() const {
    return impl->get_package_name();
}

std::string ExternalClass::get_package_name_string() const {
    return impl->get_package_name_string();
}

std::string_view ExternalClass::get_dalvik_name() const {
    return impl->get_dalvik_name();
}

std::string ExternalClass::get_dalvik_name_string() const {
    return impl->get_dalvik_name_string();
}

std::string_view ExternalClass::get_canonical_name() const {
    return impl->get_canonical_name();
}

std::string ExternalClass::get_canonical_name_string() const {
    return impl->get_canonical_name_string();
}