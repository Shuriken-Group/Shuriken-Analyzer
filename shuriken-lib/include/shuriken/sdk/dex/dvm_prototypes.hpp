//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>
#include <string_view>
#include <string>

#include "shuriken/sdk/dex/custom_types.hpp"

namespace shuriken {
namespace dex {
class DVMPrototypeProvider;
class DVMType;

class DVMPrototype {
private:
    std::reference_wrapper<DVMPrototypeProvider> dvm_prototype_provider;
public:
    // constructors & destructors
    DVMPrototype(DVMPrototypeProvider&);
    ~DVMPrototype() = default;

    DVMPrototype(const DVMPrototype&) = delete;
    DVMPrototype& operator=(const DVMPrototype&) = delete;

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
    const DVMType* get_return_type() const;

    /**
     * @return Get a pointer to the return type
     */
    DVMType * get_return_type();

    /**
     * @return an iterator to the list of parameter types from the prototype
     */
    dvmtypes_list_deref_iterator_t get_parameters();
};

}
}