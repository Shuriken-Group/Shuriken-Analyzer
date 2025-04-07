//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <shuriken/internal/engine/dex/parser/dex_header.hpp>
#include <shuriken/internal/engine/dex/parser/field_id.hpp>
#include <shuriken/internal/engine/dex/parser/method_id.hpp>


#include <shuriken/internal/io/shurikenstream.hpp>
#include <shuriken/sdk/common/error.hpp>
#include <shuriken/internal/providers/dex/dvm_types_provider.hpp>
#include <shuriken/internal/providers/dex/dvm_prototypes_provider.hpp>
#include <shuriken/sdk/dex/dvm_prototypes.hpp>
#include <shuriken/internal/providers/dex/custom_types.hpp>

namespace shuriken {
namespace dex {

class Parser {
private:
    DexHeader header_;
    std::vector<std::string> string_pool;
    std::vector<std::unique_ptr<DVMTypeProvider>> types_pool;
    std::vector<std::unique_ptr<DVMType>> dvm_types_pool;
    std::vector<std::unique_ptr<DVMPrototypeProvider>> prototypes_pool;
    std::vector<std::unique_ptr<DVMPrototype>> dvm_prototype_pool;
    std::vector<FieldID> fields_;
    std::vector<MethodID> methods_;
    
    bool parse_string_pool(io::ShurikenStream& stream,
                                        std::uint32_t strings_offset,
                                        std::uint32_t n_of_strings);

    DVMTypeProvider * parse_type(std::string_view name);

    bool parse_types(io::ShurikenStream& stream,
                     std::uint32_t types_offset,
                     std::uint32_t n_of_types);

    std::vector<dvmtype_t> parse_parameters(io::ShurikenStream& stream,
                                            std::uint32_t parameters_off);

    bool parse_protos(io::ShurikenStream& stream,
                      std::uint32_t protos_offset,
                      std::uint32_t n_of_protos);

    bool parse_fields(io::ShurikenStream& stream,
                      std::uint32_t fields_offset,
                      std::uint32_t n_of_fields);

    bool parse_methods(io::ShurikenStream& stream,
                       std::uint32_t methods_offset,
                       std::uint32_t methods_size);
public:
    Parser() = default;
    ~Parser() = default;

    error::VoidResult parse(io::ShurikenStream& stream);

    std::vector<std::unique_ptr<DVMTypeProvider>> & get_types_pool();

    std::vector<std::unique_ptr<DVMType>> & get_dvm_types_pool();

    std::vector<std::unique_ptr<DVMPrototypeProvider>> & get_prototypes_pool();

    std::vector<std::unique_ptr<DVMPrototype>> & get_dvm_prototype_pool();

    std::vector<FieldID> & get_fields_ids();

    std::vector<MethodID> & get_methods_ids();


};
}
}