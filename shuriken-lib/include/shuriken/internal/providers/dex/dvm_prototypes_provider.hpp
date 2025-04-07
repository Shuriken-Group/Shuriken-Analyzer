//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>

namespace shuriken {
namespace dex {

class DVMPrototypeProvider {
private:
    // @brief shorty_idx representing the prototype in a dalvik format
    const std::string shorty_idx;
    // @brief return type of the prototype
    dvmtype_t return_type;
    // @brief all the parameters from the prototype
    std::vector<dvmtype_t> parameter_types;
    // @brief descriptor of the prototype in dalvik format
    std::string descriptor;

public:
    DVMPrototypeProvider(std::string_view shorty_idx, DVMType& return_type, std::vector<dvmtype_t>& parameter_types);
    ~DVMPrototypeProvider() = default;

    /**
     * @return Get the shorty_idx with a string version of the prototype
     */
    std::string_view get_shorty_idx() const;

    /**
     * @return Get the shorty_idx with a string version of the prototype as a string
     */
    std::string get_shorty_idx_string() const;

    /**
     * @return Get a constant pointer to the return type
     */
    const DVMType& get_return_type() const;

    /**
     * @return Get a pointer to the return type
     */
    DVMType & get_return_type();

    /**
     * @return an iterator to the list of parameter types from the prototype
     */
    dvmtypes_list_deref_iterator_t get_parameters();

    std::string_view get_descriptor();

    std::string get_descriptor_string();
};


} // namespace dex
} // namespace shuriken