//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include <shuriken/internal/io/shurikenstream.hpp>
#include <shuriken/internal/engine/dex/parser/field_id.hpp>
#include <shuriken/internal/engine/dex/parser/method_id.hpp>
#include <shuriken/internal/engine/dex/parser/class_data_item.hpp>

#include <shuriken/sdk/common/iterator_range.hpp>
#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/dvm_types.hpp>

namespace shuriken {
namespace dex {

class ClassDef {
public:
    using interfaces_list_t = std::vector<DVMType*>;
    using it_interfaces_list = iterator_range<interfaces_list_t::iterator>;
private:
    //! idx for the current class
    std::uint32_t class_idx;
    //! flags for this class
    std::uint32_t access_flags;
    //! parent class id
    std::uint32_t superclass_idx;
    //! interfaces implemented by class
    std::uint32_t interfaces_off;
    //! idx to a string with source file
    std::uint32_t source_file_idx;
    //! debugging information and other data
    std::uint32_t annotations_off;
    //! offset to class data item
    std::uint32_t class_data_off;
    //! offset to static values
    std::uint32_t static_values_off;

    //! DVMClass for the current class
    DVMType * class_type;
    //! DVMClass for the parent/super class
    DVMType * superclass_type;
    //! source file
    std::string source_file;

    interfaces_list_t interfaces;

    /// @brief ClassDataItem value fo the current class
    ClassDataItem class_data_item;
    /// @brief Array of initial values for static fields.
    EncodedArray static_values;
public:
    ClassDef();
    ~ClassDef() = default;

    bool parse_class_def(io::ShurikenStream& stream,
                         const std::vector<std::string>& string_pools,
                         const std::vector<std::unique_ptr<DVMType>>& types_pool,
                         const std::vector<FieldID>& fields,
                         const std::vector<MethodID>& methods);

    std::uint32_t get_class_idx() const;

    std::uint32_t get_access_flags() const;

    std::uint32_t get_superclass_idx() const;

    std::uint32_t get_interfaces_off() const;

    std::uint32_t get_source_file_idx() const;

    std::uint32_t get_annotations_off() const;

    std::uint32_t get_class_data_off() const;

    std::uint32_t get_static_values_off() const;

    DVMType * get_class_type() const;

    DVMType * get_superclass_type() const;

    std::string_view get_source_file();

    std::string get_source_file_string();

    it_interfaces_list get_interfaces();
};

}
}