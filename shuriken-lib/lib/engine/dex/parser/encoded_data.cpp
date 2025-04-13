//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/internal/engine/dex/parser/encoded_data.hpp>

using namespace shuriken::dex;

void EncodedArray::parse_encoded_array(shuriken::io::ShurikenStream &stream,
                                       const std::vector<std::unique_ptr<DVMType>> &types_pool,
                                       const std::vector<std::string> &string_pools) {
    auto array_size = stream.read_uleb128();
    std::unique_ptr<EncodedValue> value = nullptr;

    for (std::uint64_t I = 0; I < array_size; ++I) {
        value = std::make_unique<EncodedValue>();
        value->parse_encoded_value(stream, types_pool, string_pools);
        values.push_back(std::move(value));
    }
}


size_t EncodedArray::get_encodedarray_size() const {
    return values.size();
}

EncodedArray::it_encoded_value EncodedArray::get_encoded_values() {
    auto &aux = get_encoded_values_vector();
    return deref_iterator_range(aux);
}

EncodedArray::it_const_encoded_value EncodedArray::get_encoded_values_const() {
    const auto &aux = get_encoded_values_vector();
    return deref_iterator_range(aux);
}

EncodedArray::encoded_values_s_t &EncodedArray::get_encoded_values_vector() {
    if (values_s.empty() || values_s.size() != values.size()) {
        values_s.clear();
        for (const auto &entry: values)
            values_s.push_back(std::ref(*entry));
    }
    return values_s;
}


void EncodedAnnotation::parse_encoded_annotation(shuriken::io::ShurikenStream &stream,
                                                 const std::vector<std::unique_ptr<DVMType>> &types_pool,
                                                 const std::vector<std::string> &string_pools) {
    std::unique_ptr<EncodedValue> value;
    std::uint64_t name_idx;
    auto type_idx = stream.read_uleb128();
    auto size = stream.read_uleb128();

    type = types_pool[type_idx].get();

    for (std::uint64_t I = 0; I < size; ++I) {
        // read first the name_idx
        name_idx = stream.read_uleb128();
        // then the EncodedValue
        value = std::make_unique<EncodedValue>();
        value->parse_encoded_value(stream, types_pool, string_pools);
        annotations.emplace_back(string_pools[name_idx], std::move(value));
    }
}

DVMType *EncodedAnnotation::get_annotation_type() {
    return type;
}

size_t EncodedAnnotation::get_number_of_annotations() const {
    return annotations.size();
}

const EncodedAnnotation::annotation_elements_t &EncodedAnnotation::get_annotations() {
    return annotations;
}


void EncodedValue::parse_encoded_value(shuriken::io::ShurikenStream &stream,
                                       const std::vector<std::unique_ptr<DVMType>> &types_pool,
                                       const std::vector<std::string> &string_pools) {

}

types::value_format EncodedValue::get_format() {
    return format;
}

shuriken::error::Result<std::reference_wrapper<EncodedValue::data_buffer_t>>
EncodedValue::get_data_buffer() {
    if (format == types::value_format::VALUE_ARRAY ||
        format == types::value_format::VALUE_ANNOTATION)
        return error::make_error<std::reference_wrapper<data_buffer_t>>(
                error::ErrorCode::InvalidType,
                "Error value does not contain a data buffer");

    // Get a reference to the data_buffer_t inside the variant
    auto& value_data = std::get<data_buffer_t>(value);

    // Return a success with the reference_wrapper
    return error::make_success(std::ref(value_data));
}


EncodedArray *EncodedValue::get_array_data() {
    if (format == types::value_format::VALUE_ARRAY)
        return std::get<std::unique_ptr<EncodedArray>>(value).get();
    return nullptr;
}

EncodedAnnotation *EncodedValue::get_annotation_data() {
    if (format == types::value_format::VALUE_ANNOTATION)
        return std::get<std::unique_ptr<EncodedAnnotation>>(value).get();
    return nullptr;
}

std::int32_t EncodedValue::convert_data_to_int() {
    if (format != types::value_format::VALUE_INT)
        return 0;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    return *(reinterpret_cast<std::int32_t *>(value_data.data()));
}

std::int64_t EncodedValue::convert_data_to_long() {
    if (format != types::value_format::VALUE_LONG)
        return 0;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    return *(reinterpret_cast<std::int64_t *>(value_data.data()));
}

std::uint8_t EncodedValue::convert_data_to_byte() {
    if (format != types::value_format::VALUE_BYTE)
        return 0;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    return *(reinterpret_cast<std::uint8_t *>(value_data.data()));
}

std::int16_t EncodedValue::convert_data_to_short() {
    if (format != types::value_format::VALUE_SHORT)
        return 0;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    return *(reinterpret_cast<std::int16_t *>(value_data.data()));
}

double EncodedValue::convert_data_to_double() {
    union long_double {
        std::uint64_t long_bits;
        double double_bits;
    };
    if (format != types::value_format::VALUE_DOUBLE)
        return 0.0;
    long_double data;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    data.long_bits = *(reinterpret_cast<std::uint64_t *>(value_data.data()));
    return data.double_bits;
}

float EncodedValue::convert_data_to_float() {
    union int_float {
        std::uint32_t int_bits;
        float float_bits;
    };
    if (format != types::value_format::VALUE_FLOAT)
        return 0.0f;
    int_float data;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    data.int_bits = *(reinterpret_cast<std::uint32_t *>(value_data.data()));
    return data.float_bits;
}

std::uint16_t EncodedValue::convert_data_to_char() {
    if (format != types::value_format::VALUE_CHAR)
        return 0;
    auto &value_data = std::get<std::vector<std::uint8_t>>(value);
    return *(reinterpret_cast<std::uint16_t *>(value_data.data()));
}

EncodedField::EncodedField(const FieldID &field, types::access_flags flags) : field(field), flags(flags) {
}

const FieldID &EncodedField::get_field() {
    return field;
}


types::access_flags EncodedField::get_flags() {
    return flags;
}


void EncodedField::set_initial_value(EncodedArray *value) {
    this->initial_value = value;
}


EncodedArray *EncodedField::get_initial_value() {
    return initial_value;
}

void EncodedCatchHandler::parse_encoded_catch_handler(shuriken::io::ShurikenStream &stream,
                                                      const std::vector<std::unique_ptr<DVMType>> &types_pool) {
    std::uint64_t type_idx, addr;

    offset = stream.position();
    size = stream.read_sleb128();


    for (size_t I = 0, S = std::abs(size); I < S; ++I) {
        type_idx = stream.read_uleb128();
        addr = stream.read_uleb128();

        handlers.push_back({.type = types_pool[type_idx].get(), .idx = addr});
    }

    // A size of 0 means that there is a catch-all but no explicitly typed catches
    // And a size of -1 means that there is one typed catch along with a catch-all.
    if (size <= 0)
        catch_all_addr = stream.read_uleb128();
}

bool EncodedCatchHandler::has_explicit_typed_catches() const {
    if (size >= 0) return true;// user should check size of handlers
    return false;
}

std::int64_t EncodedCatchHandler::get_size() const {
    return size;
}

std::uint64_t EncodedCatchHandler::get_catch_all_addr() const {
    return catch_all_addr;
}

std::uint64_t EncodedCatchHandler::get_offset() const {
    return offset;
}

EncodedCatchHandler::it_handler_pairs EncodedCatchHandler::get_handle_pairs() {
    return make_range(handlers.begin(), handlers.end());
}

void CodeItemStruct::parse_code_item_struct(shuriken::io::ShurikenStream &stream,
                                            const std::vector<std::unique_ptr<DVMType>> &types_pool) {
    // instructions are read in chunks of 16 bits
    std::uint8_t instruction[2];
    size_t I;
    std::unique_ptr<EncodedCatchHandler> encoded_catch_handler;

    registers_size = stream.read<std::uint16_t>();
    ins_size = stream.read<std::uint16_t>();
    outs_size = stream.read<std::uint16_t>();
    tries_size = stream.read<std::uint16_t>();
    debug_info_off = stream.read<std::uint32_t>();
    insns_size = stream.read<std::uint32_t>();

    // now we can work with the values

    // first read the instructions for the CodeItem
    instructions_raw.reserve(insns_size * 2);

    for (I = 0; I < insns_size; ++I) {
        // read the instruction
        stream.read_bytes(reinterpret_cast<char *>(instruction), 2);
        instructions_raw.push_back(instruction[0]);
        instructions_raw.push_back(instruction[1]);
    }

    if ((tries_size > 0) && // padding present in case tries_size > 0
        (insns_size % 2)) {// and instructions size is odd
        stream.seek(sizeof(std::uint16_t), std::ios_base::cur);
    }

    if (tries_size > 0) {
        TryItem try_item = {0, 0, 0};

        for (I = 0; I < tries_size; ++I) {
            stream.read_bytes(reinterpret_cast<char *>(&try_item), sizeof(TryItem));
            try_items.push_back(try_item);
        }

        std::uint64_t encoded_catch_handler_size = stream.read_uleb128();
        for (I = 0; I < encoded_catch_handler_size; I++) {
            encoded_catch_handler = std::make_unique<EncodedCatchHandler>();
            encoded_catch_handler->parse_encoded_catch_handler(stream, types_pool);
            encoded_catch_handlers.push_back(std::move(encoded_catch_handler));
        }
    }
}


std::uint16_t CodeItemStruct::get_registers_size() const {
    return registers_size;
}

std::uint16_t CodeItemStruct::get_incomings_args() const {
    return ins_size;
}

std::uint16_t CodeItemStruct::get_outgoing_args() const {
    return outs_size;
}

std::uint16_t CodeItemStruct::get_number_try_items() const {
    return tries_size;
}

std::uint32_t CodeItemStruct::get_offset_to_debug_info() const {
    return debug_info_off;
}

std::uint32_t CodeItemStruct::get_instructions_size() const {
    return insns_size;
}

std::vector<std::uint8_t> &CodeItemStruct::get_bytecode() {
    return instructions_raw;
}


CodeItemStruct::it_try_items CodeItemStruct::get_try_items() {
    return make_range(try_items.begin(), try_items.end());
}


CodeItemStruct::it_encoded_catch_handlers CodeItemStruct::get_encoded_catch_handlers() {
    if (encoded_catch_handlers_s.empty())
        for (const auto &encoded_catch_handler: encoded_catch_handlers)
            encoded_catch_handlers_s.push_back(*encoded_catch_handler.get());
    return deref_iterator_range(encoded_catch_handlers_s);
}

CodeItemStruct::encoded_catch_handlers_t &CodeItemStruct::get_encoded_catch_handlers_vector() {
    return encoded_catch_handlers;
}

EncodedMethod::EncodedMethod(const MethodID &method_id, types::access_flags access_flags) : method_id(method_id),
                                                                                            access_flags(access_flags) {
}


void EncodedMethod::parse_encoded_method(shuriken::io::ShurikenStream &stream, std::uint64_t code_off,
                                         const std::vector<std::unique_ptr<DVMType>> &types_pool) {
    auto current_offset = stream.position();

    if (code_off > 0) {
        stream.seek(code_off);
        // parse the code item
        code_item = std::make_unique<CodeItemStruct>();
        code_item->parse_code_item_struct(stream, types_pool);
    }

    stream.seek(current_offset);
}

const MethodID &EncodedMethod::get_method_id() {
    return method_id;
}


types::access_flags EncodedMethod::get_access_flags() {
    return access_flags;
}


CodeItemStruct *EncodedMethod::get_code_items() {
    return code_item.get();
}
