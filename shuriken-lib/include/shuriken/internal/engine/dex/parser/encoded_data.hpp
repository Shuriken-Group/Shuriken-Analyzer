//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>

#include <shuriken/internal/io/shurikenstream.hpp>

#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>
#include <shuriken/sdk/common/iterator_range.hpp>
#include <shuriken/sdk/common/deref_iterator_range.hpp>

namespace shuriken {
namespace dex {

class MethodID;
class FieldID;
class EncodedValue;

class EncodedArray {
public:
    using encoded_values_t = std::vector<std::unique_ptr<EncodedValue>>;
    using encoded_values_s_t = std::vector<std::reference_wrapper<EncodedValue>>;
    using it_encoded_value = deref_iterator_range<encoded_values_s_t>;
    using it_const_encoded_value = deref_iterator_range<
            const encoded_values_s_t>;
private:
    encoded_values_t values;
    encoded_values_s_t values_s;
public:
    EncodedArray() = default;
    ~EncodedArray() = default;

    void parse_encoded_array(io::ShurikenStream& stream,
                             const std::vector<std::unique_ptr<DVMType>>& types_pool,
                             const std::vector<std::string>& string_pools);

    size_t get_encodedarray_size() const;

    it_encoded_value get_encoded_values();

    it_const_encoded_value get_encoded_values_const();

    encoded_values_s_t &get_encoded_values_vector();
};

struct AnnotationElement {
    std::string name;
    std::unique_ptr<EncodedValue> value;
};

class EncodedAnnotation {
public:
    using annotation_elements_t = std::vector<AnnotationElement>;
private:
    DVMType * type;
    annotation_elements_t annotations;
public:
    EncodedAnnotation() = default;
    ~EncodedAnnotation() = default;

    void parse_encoded_annotation(io::ShurikenStream& stream,
                             const std::vector<std::unique_ptr<DVMType>>& types_pool,
                             const std::vector<std::string>& string_pools);

    DVMType * get_annotation_type();

    size_t get_number_of_annotations() const;

    const annotation_elements_t & get_annotations();
};

class EncodedValue {
public:
    using data_buffer_t = std::vector<std::uint8_t>;
private:
    types::value_format format;
    std::variant<data_buffer_t,
                    std::unique_ptr<EncodedArray>,
                    std::unique_ptr<EncodedAnnotation>> value;
public:
    EncodedValue() = default;
    ~EncodedValue() = default;

    void parse_encoded_value(io::ShurikenStream& stream,
                                  const std::vector<std::unique_ptr<DVMType>>& types_pool,
                                  const std::vector<std::string>& string_pools);

    types::value_format get_format();

    error::Result<std::reference_wrapper<data_buffer_t>> get_data_buffer();

    EncodedArray *get_array_data();

    EncodedAnnotation *get_annotation_data();

    std::int32_t convert_data_to_int();

    std::int64_t convert_data_to_long();

    std::uint8_t convert_data_to_byte();

    std::int16_t convert_data_to_short();

    double convert_data_to_double();

    float convert_data_to_float();

    std::uint16_t convert_data_to_char();
};

class EncodedField {
private:
    std::reference_wrapper<const FieldID> field;
    types::access_flags flags;
    EncodedArray * initial_value;
public:
    EncodedField(const FieldID & field, types::access_flags flags);
    ~EncodedField() = default;

    const FieldID & get_field();

    types::access_flags get_flags();

    void set_initial_value(EncodedArray * value);

    EncodedArray *get_initial_value();
};

struct EncodedTypePair {
    //! Type catched by exception
    DVMType * type;
    //! idx where the exception is catched
    std::uint64_t idx;
};

struct TryItem {
    //! start address of the block of code covered by this entry
    std::uint32_t start_addr;
    //! number of 16-bit code units covered by this entry
    std::uint16_t insn_count;
    //! offset in bytes from starts of associated encoded_catch_handler_list
    std::uint16_t handler_off;
};

class EncodedCatchHandler {
public:
    using handler_pairs_t = std::vector<EncodedTypePair>;
    using it_handler_pairs = iterator_range<handler_pairs_t::iterator>;
private:
    /// @brief Size of the vector of EncodedTypePair
    /// if > 0 indicates the size of the handlers
    /// if == 0 there are no handlers nor catch_all_addr
    /// if < 0 no handlers and catch_all_addr is set
    std::int64_t size;
    /// @brief vector of encoded type pair
    handler_pairs_t handlers;
    /// @brief bytecode of the catch all-handler.
    /// This element is only present if size is non-positive.
    std::uint64_t catch_all_addr = 0;
    /// @brief Offset where the encoded catch handler is
    /// in the file
    std::streampos offset;
public:
    /// @brief Constructor of EncodedCatchHandler
    EncodedCatchHandler() = default;
    /// @brief Destructor of EncodedCatchHandler
    ~EncodedCatchHandler() = default;

    void parse_encoded_catch_handler(io::ShurikenStream& stream,
                                     const std::vector<std::unique_ptr<DVMType>>& types_pool);

    /// @brief Check value of size to test if there are encodedtypepairs
    /// @return if there are explicit typed catches
    bool has_explicit_typed_catches() const;

    /// @brief Get the size of the EncodedCatchHandler
    /// @return value of size, refer to `size` documentation
    /// to check the possible values
    std::int64_t get_size() const;

    /// @brief Return the value from catch_all_addr
    /// @return catch_all_addr value
    std::uint64_t get_catch_all_addr() const;

    /// @brief Get the offset where encoded catch handler is
    /// @return offset of encoded catch handler
    std::uint64_t get_offset() const;

    /// @brief Get an iterator to the handle pairs
    /// @return iterator to handlers
    it_handler_pairs get_handle_pairs();
};

class CodeItemStruct {
public:
    using try_items_t = std::vector<TryItem>;
    using it_try_items = iterator_range<try_items_t::iterator>;

    using encoded_catch_handlers_t = std::vector<std::unique_ptr<EncodedCatchHandler>>;
    using encoded_catch_handlers_s_t = std::vector<std::reference_wrapper<EncodedCatchHandler>>;
    using it_encoded_catch_handlers = deref_iterator_range<encoded_catch_handlers_s_t>;
private:
    std::uint16_t registers_size;//! number of registers used in the code
    std::uint16_t ins_size;      //! number of words of incoming arguments to the method
    std::uint16_t outs_size;     //! number of words of outgoung arguments space required
    //! for method invocation.
    std::uint16_t tries_size;    //! number of TryItem, can be 0
    std::uint32_t debug_info_off;//! offset to debug_info_item
    std::uint32_t insns_size;    //! size of instruction list

    std::vector<std::uint8_t> instructions_raw;

    try_items_t try_items;

    encoded_catch_handlers_t encoded_catch_handlers;
    encoded_catch_handlers_s_t encoded_catch_handlers_s;
public:
    CodeItemStruct() = default;
    ~CodeItemStruct() = default;
    void parse_code_item_struct(io::ShurikenStream &stream,
                                const std::vector<std::unique_ptr<DVMType>>& types_pool);

    std::uint16_t get_registers_size() const;

    std::uint16_t get_incomings_args() const;

    std::uint16_t get_outgoing_args() const;

    std::uint16_t get_number_try_items() const;

    std::uint32_t get_offset_to_debug_info() const;

    std::uint32_t get_instructions_size() const;

    std::vector<std::uint8_t> & get_bytecode();

    it_try_items get_try_items();

    it_encoded_catch_handlers get_encoded_catch_handlers();

    encoded_catch_handlers_t &get_encoded_catch_handlers_vector();

};


class EncodedMethod {
private:
    std::reference_wrapper<const MethodID> method_id;
    types::access_flags access_flags;
    std::unique_ptr<CodeItemStruct> code_item;
public:
    EncodedMethod(const MethodID & method_id, types::access_flags access_flags);
    ~EncodedMethod() = default;

    void parse_encoded_method(io::ShurikenStream& stream,
                              std::uint64_t code_off,
                              const std::vector<std::unique_ptr<DVMType>>& types_pool);

    const MethodID & get_method_id();

    types::access_flags get_access_flags();

    CodeItemStruct * get_code_items();
};

}
}