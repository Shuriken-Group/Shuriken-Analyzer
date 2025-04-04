//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>
#include <string_view>
#include <string>

namespace shuriken {
namespace dex {

class DexExternalMethodProvider;

class ExternalMethod {
private:
    std::reference_wrapper<DexExternalMethodProvider> dex_external_method_provider;
public:
    /**
    * @brief Get the name of the class that owns this external method as a string_view
    * @return String view of the owner class name
    */
    std::string_view get_class_name() const;

    /**
     * @brief Get the name of the class that owns this external method as a string
     * @return String copy of the owner class name
     */
    std::string get_class_name_string() const;

    /**
     * @brief Get the name of this external method as a string_view
     * @return String view of the method name
     */
    std::string_view get_name() const;

    /**
     * @brief Get the name of this external method as a string
     * @return String copy of the method name
     */
    std::string get_name_string() const;

    /***
     * @return a view of method's descriptor
     * package_name/class_name->method_name(params)retType
     */
    std::string_view get_descriptor() const;

    /***
     * @return a string of method's descriptor
     * package_name/class_name->method_name(params)retType
     */
    std::string get_descriptor_string() const;
};


} // namespace dex
} // namespace shuriken