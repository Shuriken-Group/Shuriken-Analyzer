//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include "shuriken/sdk/dex/custom_types.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"

namespace shuriken {
namespace dex {

/**
 * @brief Represents a field identifier from the DEX file's field_ids section
 * 
 * FieldID objects contain the basic identifying information for fields:
 * the class that contains the field, the field's type, and the field name.
 * This is used to uniquely identify fields within and across DEX files.
 */
class FieldID {
private:
    std::reference_wrapper<DVMType> class_;
    std::reference_wrapper<DVMType> type_;
    std::string name_;
public:
    /**
     * @brief Construct a new FieldID object
     * @param class_ Reference to the class type that owns this field
     * @param type_ Reference to the field's type
     * @param name_ The field name
     */
    FieldID(DVMType& class_, DVMType& type_, std::string_view name_);
    ~FieldID() = default;

    /**
     * @brief Get the class type that owns this field
     * @return Reference to the DVMType representing the owner class
     */
    DVMType & get_class();

    /**
     * @brief Get the field's type
     * @return Reference to the DVMType representing the field type
     */
    DVMType & get_type();

    /**
     * @brief Get the field name as a string view
     * @return String view of the field name
     */
    std::string_view get_name();

    /**
     * @brief Get the field name as a string
     * @return String copy of the field name
     */
    std::string get_name_string();
};
}
}