//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/class.hpp"
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/sdk/dex/constants.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"
#include "shuriken/sdk/dex/disassembly_constants.hpp"
#include "shuriken/sdk/dex/instruction.hpp"
#include "shuriken/sdk/dex/control_flow_graph.hpp"
#include "shuriken/internal/sdk/dex/control_flow_graph_impl.hpp"


#include "shuriken/internal/engine/dex/dex_engine.hpp"

#include <iostream>
#include <vector>
#include <span>
#include <memory>
#include <list>



namespace shuriken::dex {

class Dex;
class EncodedMethod;

namespace {
    // Template factory function for creating instruction wrappers
    template<typename WrapperType, typename ProviderType>
    std::unique_ptr<Instruction> get_instruction(ProviderType &provider) {
        return std::make_unique<WrapperType>(provider);
    }

    // Type alias for the factory function
    using instruction_factory_func = std::function<std::unique_ptr<Instruction>(void *)>;

    // Wrapper function to handle the void* casting
    template<typename WrapperType, typename ProviderType>
    std::unique_ptr<Instruction> create_wrapper(void *provider_ptr) {
        auto *typed_provider = static_cast<ProviderType *>(provider_ptr);
        return std::make_unique<WrapperType>(*typed_provider);
    }
}


class Method::Impl {
private:
    std::reference_wrapper<DexEngine> dex_engine;
    // @brief name of the method
    std::string method_name;
    // @brief access flags from the method
    types::access_flags method_access_flags;
    // @brief string with the access flags from the method as string
    std::string access_flags_str;
    // @brief method type taken from the generation of the class
    types::method_type_e method_type;
    // @brief pointer to the prototype of the method
    std::reference_wrapper<DVMPrototype> method_prototype;
    // @brief Pointer to owner class (it can be nullptr)
    std::reference_wrapper<Class> owner_class;
    // @brief Pointer to owner Dex (it can be nullptr)
    std::reference_wrapper<Dex> owner_dex;
    // @brief descriptor of the method
    std::string method_descriptor;
    // @brief number of registers used in the op_codes
    std::uint16_t number_of_registers;
    // @brief span that points to the op_codes
    std::vector<std::uint8_t> bytecode;
    // @brief pointer to the EncodedMethod to extract information
    EncodedMethod * method;
    // @brief unique pointer to the control flow graph structure
    // generated with the instructions from the method
    std::unique_ptr<ControlFlowGraph> control_flow_graph;

    // @brief flag to know
    bool disassembled = false;

    // @brief List of instructions for the method
    std::list<std::unique_ptr<Instruction>> instructions;
    std::list<std::reference_wrapper<Instruction>> instructions_r;

    // @brief List of exceptions for the method
    disassembler::exceptions_data_t exceptions;

public:
    Impl(std::string_view name,
         types::access_flags access_flags,
         DVMPrototype &method_prototype,
         types::method_type_e method_type,
         Class &owner_class,
         Dex &owner_dex,
         DexEngine &dex_engine,
         std::uint16_t number_of_registers,
         std::vector<std::uint8_t> &bytecode,
         EncodedMethod *method)
            : method_name(name),
              method_access_flags(access_flags), method_prototype(method_prototype),
              method_type(method_type),
              owner_class(owner_class), owner_dex(owner_dex),
              dex_engine(dex_engine),
              number_of_registers(number_of_registers),
              bytecode(std::move(bytecode)),
              method(method) {
        method_descriptor = owner_class.get_name_string() + "->"
                            + std::string(name) + method_prototype.get_descriptor_string();
    }

    ~Impl() = default;

    std::string_view get_name() const {
        return method_name;
    }


    std::string get_name_string() const {
        return method_name;
    }


    shuriken::dex::types::access_flags get_method_access_flags() const {
        return method_access_flags;
    }

    std::string_view get_method_access_flags_str() {
        if (access_flags_str.empty()) {
            access_flags_str = access_flags_to_string(method_access_flags);
        }
        return access_flags_str;
    }

    const shuriken::dex::DVMPrototype &get_method_prototype() const {
        return method_prototype;
    }


    shuriken::dex::DVMPrototype &get_method_prototype() {
        return method_prototype;
    }

    types::method_type_e get_method_type() const {
        return method_type;
    }


    const shuriken::dex::Class &get_owner_class() const {
        return owner_class;
    }


    shuriken::dex::Class &get_owner_class() {
        return owner_class;
    }

    const shuriken::dex::Dex &get_owner_dex() const {
        return owner_dex;
    }


    shuriken::dex::Dex &get_owner_dex() {
        return owner_dex;
    }


    std::string_view get_descriptor() const {
        return method_descriptor;
    }


    std::string get_descriptor_string() const {
        return method_descriptor;
    }


    std::uint16_t registers_size() const {
        return number_of_registers;
    }


    std::span<std::uint8_t> get_bytecode() {
        return std::span<std::uint8_t>(bytecode);
    }

    std::vector<std::uint8_t> &get_bytecode_vector() {
        return bytecode;
    }

    EncodedMethod *get_encoded_method() const {
        return method;
    }

    ControlFlowGraph& get_control_flow_graph() {
        if (control_flow_graph == nullptr)
            this->dex_engine.get().generate_cfgf(*this);
        return *control_flow_graph.get();
    }

    void set_method_instructions(std::list<std::unique_ptr<Instruction>> &insns) {
        this->instructions = std::move(insns);
    }

    void set_exceptions(disassembler::exceptions_data_t& exceptionsData) {
        this->exceptions = std::move(exceptionsData);
    }

    void set_control_flow_graph(std::unique_ptr<ControlFlowGraph> & cfg) {
        this->control_flow_graph = std::move(cfg);
    }

    std::list<std::reference_wrapper<Instruction>> & get_method_instructions() {
        if (!disassembled) {
            this->dex_engine.get().disassemble_method(*this);
            disassembled = true;
        }
        if (instructions_r.empty()) {
            for (auto & instr : instructions)
                instructions_r.push_back(*instr);
        }
        return instructions_r;
    }

    std::list<std::reference_wrapper<Instruction>> get_instructions_in_range(std::uint64_t start_address, std::uint64_t end_address) {
        std::list<std::reference_wrapper<Instruction>> result;
        
        auto & instructions_ref = get_method_instructions();
        
        for (auto & instr : instructions_ref) {
            std::uint64_t instr_address = instr.get().get_address();
            if (instr_address >= start_address && instr_address <= end_address) {
                result.push_back(instr);
            }
        }
        
        return result;
    }

    disassembler::exceptions_data_t& get_exceptions() {
        if (!disassembled) {
            this->dex_engine.get().disassemble_method(*this);
            disassembled = true;
        }
        return exceptions;
    }
};

}