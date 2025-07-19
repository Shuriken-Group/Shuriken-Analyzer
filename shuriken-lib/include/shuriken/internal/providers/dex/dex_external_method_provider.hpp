//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>

namespace shuriken {
namespace dex {

class DexEngine;

class DexExternalMethodProvider {
private:
    std::reference_wrapper<DexEngine> dex_engine;
    // @brief name of the method without any type, or class name
    std::string method_name;
    // @brief descriptor name
    std::string descriptor;
    // @brief class name
    std::string class_name;
public:
    DexExternalMethodProvider(
            std::string_view method_name,
            std::string_view descriptor,
            std::string_view class_name,
            DexEngine& dex_engine
    );

    ~DexExternalMethodProvider() = default;

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

}
}