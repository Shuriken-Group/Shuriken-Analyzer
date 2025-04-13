//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/internal/engine/dex/parser/class_data_item.hpp>

using namespace shuriken::dex;

void ClassDataItem::parse_class_data_item(shuriken::io::ShurikenStream &stream,
                                          const std::vector<std::unique_ptr<DVMType>> &types_pool,
                                          const std::vector<FieldID> &fields, const std::vector<MethodID> &methods) {
    auto current_offset = stream.position();

    std::uint64_t I;
    // IDs for the different variables
    std::uint64_t static_field = 0, instance_field = 0, direct_method = 0, virtual_method = 0;
    std::uint64_t access_flags;// access flags of the variables
    std::uint64_t code_offset; // offset for parsing

    // read the sizes of the different variables
    std::uint64_t const static_fields_size = stream.read_uleb128();
    std::uint64_t const instance_fields_size = stream.read_uleb128();
    std::uint64_t const direct_methods_size = stream.read_uleb128();
    std::uint64_t const virtual_methods_size = stream.read_uleb128();

    for (I = 0; I < static_fields_size; ++I) {
        //! value needs to be incremented with the
        //! uleb128 read, so we always have that
        //! static_field = prev + uleb128
        static_field += stream.read_uleb128();
        //! now read the access flags
        access_flags = stream.read_uleb128();
        //! create the static field
        static_fields.push_back(
                std::make_unique<EncodedField>(
                        fields[static_field],
                        static_cast<types::access_flags>(access_flags)
                )
        );
    }

    for (I = 0; I < instance_fields_size; ++I) {
        instance_field += stream.read_uleb128();
        access_flags = stream.read_uleb128();
        instance_fields.push_back(
                std::make_unique<EncodedField>(
                        fields[instance_field],
                        static_cast<types::access_flags>(access_flags)
                )
        );
    }

    for (I = 0; I < direct_methods_size; ++I) {
        direct_method += stream.read_uleb128();
        access_flags = stream.read_uleb128();
        // for the code item
        code_offset = stream.read_uleb128();
        direct_methods.push_back(
                std::make_unique<EncodedMethod>(
                        methods[direct_method],
                        static_cast<types::access_flags>(access_flags)
                )
        );
        direct_methods.back()->parse_encoded_method(stream, code_offset, types_pool);
    }

    for (I = 0; I < virtual_methods_size; ++I) {
        virtual_method += stream.read_uleb128();
        access_flags = stream.read_uleb128();
        code_offset = stream.read_uleb128();
        virtual_methods.push_back(
                std::make_unique<EncodedMethod>(
                        methods[virtual_method],
                        static_cast<types::access_flags>(access_flags)
                )
        );
        virtual_methods.back()->parse_encoded_method(stream, code_offset, types_pool);
    }


    stream.seek(current_offset);
}

std::size_t ClassDataItem::get_number_of_static_fields() const {
    return static_fields.size();
}

std::size_t ClassDataItem::get_number_of_instance_fields() const {
    return instance_fields.size();
}

std::size_t ClassDataItem::get_number_of_direct_methods() const {
    return direct_methods.size();
}

std::size_t ClassDataItem::get_number_of_virtual_methods() const {
    return virtual_methods.size();
}

EncodedField *ClassDataItem::get_static_field_by_id(std::uint32_t id) {
    if (id >= static_fields.size())
        return nullptr;
    return static_fields[id].get();
}

EncodedField *ClassDataItem::get_instance_field_by_id(std::uint32_t id) {
    if (id >= instance_fields.size())
        return nullptr;
    return instance_fields[id].get();
}

EncodedMethod *ClassDataItem::get_direct_method_by_id(std::uint32_t id) {
    if (id >= direct_methods.size())
        return nullptr;
    return direct_methods[id].get();
}

EncodedMethod *ClassDataItem::get_virtual_method_by_id(std::uint32_t id) {
    if (id >= virtual_methods.size())
        return nullptr;
    return virtual_methods[id].get();
}

ClassDataItem::it_encoded_fields ClassDataItem::get_static_fields() {
    auto &aux = get_static_fields_vector();
    return deref_iterator_range(aux);
}

ClassDataItem::encoded_fields_s_t &ClassDataItem::get_static_fields_vector() {
    if (static_fields_s.empty() || static_fields_s.size() != static_fields.size()) {
        static_fields_s.clear();
        for (const auto &entry: static_fields)
            static_fields_s.push_back(std::ref(*entry));
    }
    return static_fields_s;
}

ClassDataItem::it_encoded_fields ClassDataItem::get_instance_fields() {
    auto &aux = get_instance_fields_vector();
    return deref_iterator_range(aux);
}

ClassDataItem::encoded_fields_s_t &ClassDataItem::get_instance_fields_vector() {
    if (instance_fields_s.empty() || instance_fields.size() != instance_fields_s.size()) {
        instance_fields_s.clear();
        for (const auto &entry: instance_fields)
            instance_fields_s.push_back(std::ref(*entry));
    }
    return instance_fields_s;
}

ClassDataItem::it_encoded_method ClassDataItem::get_direct_methods() {
    auto &aux = get_direct_methods_vector();
    return deref_iterator_range(aux);
}

ClassDataItem::encoded_methods_s_t &ClassDataItem::get_direct_methods_vector() {
    if (direct_methods_s.empty() || direct_methods_s.size() != direct_methods.size()) {
        direct_methods_s.clear();
        for (const auto &entry: direct_methods)
            direct_methods_s.push_back(std::ref(*entry));
    }
    return direct_methods_s;
}

ClassDataItem::it_encoded_method ClassDataItem::get_virtual_methods() {
    auto &aux = get_virtual_methods_s();
    return deref_iterator_range(aux);
}

ClassDataItem::encoded_methods_s_t &ClassDataItem::get_virtual_methods_s() {
    if (virtual_methods_s.empty() || virtual_methods_s.size() != virtual_methods.size()) {
        virtual_methods_s.clear();
        for (const auto &entry: virtual_methods)
            virtual_methods_s.push_back(std::ref(*entry));
    }
    return virtual_methods_s;
}