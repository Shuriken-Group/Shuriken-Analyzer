//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include "shuriken/sdk/dex/custom_types.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"

namespace shuriken {
namespace dex {

/**
 * @brief Represents a method identifier from the DEX file's method_ids section
 * 
 * MethodID objects contain the basic identifying information for methods:
 * the class that contains the method, the method's prototype (signature),
 * and the method name. This is used to uniquely identify methods within
 * and across DEX files.
 */
class MethodID {
private:
    std::reference_wrapper<DVMType> class_;
    std::reference_wrapper<DVMPrototype> proto_id_;
    std::string name_;
public:
    /**
     * @brief Construct a new MethodID object
     * @param class_ Reference to the class type that owns this method
     * @param proto_id_ Reference to the method's prototype/signature
     * @param name_ The method name
     */
    MethodID(DVMType& class_, DVMPrototype& proto_id_, std::string_view name_);
    ~MethodID() = default;

    /**
     * @brief Get the class type that owns this method
     * @return Reference to the DVMType representing the owner class
     */
    DVMType & get_class();

    /**
     * @brief Get the method's prototype/signature
     * @return Reference to the DVMPrototype
     */
    DVMPrototype & get_prototype();

    /**
     * @brief Get the method name as a string view
     * @return String view of the method name
     */
    std::string_view get_name();

    /**
     * @brief Get the method name as a string
     * @return String copy of the method name
     */
    std::string get_name_string();
};

}
}