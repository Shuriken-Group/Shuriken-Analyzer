//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <shuriken/internal/engine/dex/parser/encoded_data.hpp>
#include <shuriken/internal/engine/dex/parser/field_id.hpp>
#include <shuriken/internal/engine/dex/parser/method_id.hpp>

namespace shuriken {
namespace dex {

class ClassDataItem {
public:
    using encoded_fields_t = std::vector<std::unique_ptr<EncodedField>>;
    using encoded_fields_s_t = std::vector<std::reference_wrapper<EncodedField>>;
    using it_encoded_fields = deref_iterator_range<encoded_fields_s_t>;

    using encoded_methods_t = std::vector<std::unique_ptr<EncodedMethod>>;
    using encoded_methods_s_t = std::vector<std::reference_wrapper<EncodedMethod>>;
    using it_encoded_method = deref_iterator_range<encoded_methods_s_t>;
private:
    /// @brief Static fields from the class
    encoded_fields_t static_fields;
    encoded_fields_s_t static_fields_s;

    /// @brief Instance fields from the class
    encoded_fields_t instance_fields;
    encoded_fields_s_t instance_fields_s;

    /// @brief Direct methods from the class
    encoded_methods_t direct_methods;
    encoded_methods_s_t direct_methods_s;

    /// @brief Virtual methods from the class
    encoded_methods_t virtual_methods;
    encoded_methods_s_t virtual_methods_s;

public:
    ClassDataItem() = default;
    ~ClassDataItem() = default;

    void parse_class_data_item(io::ShurikenStream & stream,
                               const std::vector<std::unique_ptr<DVMType>>& types_pool,
                               const std::vector<FieldID>& fields,
                               const std::vector<MethodID>& methods);


    std::size_t get_number_of_static_fields() const;

    std::size_t get_number_of_instance_fields() const;

    std::size_t get_number_of_direct_methods() const;

    std::size_t get_number_of_virtual_methods() const;

    EncodedField *get_static_field_by_id(std::uint32_t id);

    EncodedField *get_instance_field_by_id(std::uint32_t id);

    EncodedMethod *get_direct_method_by_id(std::uint32_t id);

    EncodedMethod *get_virtual_method_by_id(std::uint32_t id);

    it_encoded_fields get_static_fields();

    encoded_fields_s_t &get_static_fields_vector();

    it_encoded_fields get_instance_fields();

    encoded_fields_s_t &get_instance_fields_vector();

    it_encoded_method get_direct_methods();

    encoded_methods_s_t &get_direct_methods_vector();

    it_encoded_method get_virtual_methods();

    encoded_methods_s_t &get_virtual_methods_s();
};


}
}