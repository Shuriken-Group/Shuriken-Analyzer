//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/engine/dex/disassembler/internal_disassembler.hpp"
#include "shuriken/internal/engine/dex/parser/encoded_data.hpp"
#include "shuriken/internal/sdk/dex/instruction_impl.hpp"

#include <memory>

using namespace shuriken::dex;

namespace {
    typedef std::unique_ptr<Instruction> (*generator_func)(std::span<std::uint8_t>, std::size_t, DexEngine &);

    typedef std::unique_ptr<Instruction> (*generator_func_no_dex)(std::span<std::uint8_t>, std::size_t);

    template<std::derived_from<Instruction> T>
    std::unique_ptr<Instruction>
    get_instruction(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) {
        static_assert(std::is_base_of_v<Instruction, T>, "T must be derived from Instruction");
        typename T::Impl *impl = new typename T::Impl(bytecode, index, dex);
        return std::make_unique<T>(impl);
    }

    template<std::derived_from<Instruction> T>
    std::unique_ptr<Instruction>
    get_instruction_no_dex(std::span<uint8_t> bytecode, std::size_t index) {
        static_assert(std::is_base_of_v<Instruction, T>, "T must be derived from Instruction");
        typename T::Impl *impl = new typename T::Impl(bytecode, index);
        return std::make_unique<T>(impl);
    }

    std::unordered_map<disassembler::opcodes, std::pair<generator_func, generator_func_no_dex>> function_pointers = {
#define INSTRUCTION_MAPPING(OPCODE, PROVIDER_CLASS) \
        {disassembler::opcodes::OPCODE, {&get_instruction<PROVIDER_CLASS>, &get_instruction_no_dex<PROVIDER_CLASS>}},

#include "definitions/instruction_mappings.def"

#undef INSTRUCTION_MAPPING
    };

    std::unique_ptr<Instruction>
    get_instruction_generic(disassembler::opcodes opcode, std::span<uint8_t> bytecode, std::size_t index,
                            DexEngine *dex_engine) {
        if (function_pointers.find(opcode) == function_pointers.end()) {
            DalvikIncorrectInstruction::Impl *impl = new DalvikIncorrectInstruction::Impl(bytecode, index,
                                                                                          "Error, opcode not recognized",
                                                                                          1, index, opcode);
            return std::make_unique<DalvikIncorrectInstruction>(impl);
        }
        return dex_engine == nullptr ? function_pointers[opcode].second(bytecode, index) :
               function_pointers[opcode].first(bytecode, index, *dex_engine);
    }

}

InternalDisassembler::InternalDisassembler(DexEngine *dex_engine) : dex_engine(dex_engine) {
}

std::unique_ptr<Instruction> InternalDisassembler::disassemble_instruction(
        disassembler::opcodes opcode,
        std::span<uint8_t> bytecode,
        std::size_t index) {
    std::unique_ptr<Instruction> instr = nullptr;

    if (disassembler::opcodes::OP_NOP == opcode) {
        auto second_opcode = bytecode[index + 1];

        if (second_opcode == 0x03) { // filled-array-data
            instr = dex_engine == nullptr ? ::get_instruction_no_dex<FillArrayData>(bytecode, index) :
                    ::get_instruction<FillArrayData>(bytecode, index, *dex_engine);
        } else if (second_opcode == 0x01) { // packed-switch-data
            instr = dex_engine == nullptr ? ::get_instruction_no_dex<PackedSwitch>(bytecode, index) :
                    ::get_instruction<PackedSwitch>(bytecode, index, *dex_engine);
        } else if (second_opcode == 0x02) { // sparse-switch-data
            instr = dex_engine == nullptr ? ::get_instruction_no_dex<SparseSwitch>(bytecode, index) :
                    ::get_instruction<SparseSwitch>(bytecode, index, *dex_engine);
        } else {
            instr = ::get_instruction_generic(opcode, bytecode, index, dex_engine);
        }
    } else {
        instr = ::get_instruction_generic(opcode, bytecode, index, dex_engine);
    }

    if (instr)
        last_instr = instr.get();

    return instr;
}

std::vector<std::int64_t> InternalDisassembler::determine_next(Instruction *instruction,
                                                               std::uint64_t curr_idx) {
    if (!instruction) return {};

    auto op_code = instruction->get_instruction_opcode();

    if ((op_code >= disassembler::opcodes::OP_GOTO) && (op_code <= disassembler::opcodes::OP_GOTO_32)) {
        std::int32_t offset = 0;

        if (op_code == disassembler::opcodes::OP_GOTO) {
            auto *goto_instr = reinterpret_cast<Instruction10t *>(instruction);
            offset = goto_instr->getNAA();
        } else if (op_code == disassembler::opcodes::OP_GOTO_16) {
            auto *goto_instr = reinterpret_cast<Instruction20t *>(instruction);
            offset = goto_instr->getNAAAA();
        } else if (op_code == disassembler::opcodes::OP_GOTO_32) {
            auto *goto_instr = reinterpret_cast<Instruction30t *>(instruction);
            offset = goto_instr->getNAAAAAAAA();
        }

        return {(offset * 2) + static_cast<std::int64_t>(curr_idx)};
    }

    if (op_code >= disassembler::opcodes::OP_IF_EQ &&
        op_code <= disassembler::opcodes::OP_IF_LEZ) {
        std::int32_t offset = 0;

        if (op_code >= disassembler::opcodes::OP_IF_EQ &&
            op_code <= disassembler::opcodes::OP_IF_LE) {
            auto *if_instr = reinterpret_cast<Instruction22t *>(instruction);
            offset = if_instr->getNCCCC();
        } else if (op_code >= disassembler::opcodes::OP_IF_EQZ &&
                   op_code <= disassembler::opcodes::OP_IF_LEZ) {
            auto *if_instr = reinterpret_cast<Instruction21t *>(instruction);
            offset = if_instr->getNBBBB();
        }

        return {
                static_cast<std::int64_t>(curr_idx) + instruction->get_instruction_length(), // fallthrough
                static_cast<std::int64_t>(curr_idx) + (offset * 2)                           // target of the jump
        };
    }

    // finally the switch instructions will have multiple
    // targets, including the one after the instruction
    if (op_code == disassembler::opcodes::OP_PACKED_SWITCH ||
        op_code == disassembler::opcodes::OP_SPARSE_SWITCH) {
        std::vector<std::int64_t> x = {static_cast<std::int64_t>(curr_idx) +
                                       instruction->get_instruction_length()};

        auto *switch_instr = reinterpret_cast<Instruction31t *>(instruction);

        switch (switch_instr->get_type_of_switch()) {
            case disassembler::PACKED_SWITCH: {
                auto *packed_switch = std::get<PackedSwitch *>(switch_instr->get_switch());
                const auto &targets = packed_switch->get_targets();

                for (auto &target: targets)
                    x.push_back(curr_idx + target * 2);
            }
                break;
            case disassembler::SPARSE_SWITCH: {
                auto *sparse_switch = std::get<SparseSwitch *>(switch_instr->get_switch());
                const auto &targets = sparse_switch->get_keys_targets();

                for (auto &key_target: targets)
                    x.push_back(curr_idx + key_target.second * 2);
            }
                break;
            default:
                return {};
        }

        return x;
    }


    // no other case, only the fallthrough of the instruction
    return {static_cast<int64_t>(curr_idx + instruction->get_instruction_length())};
}

std::vector<std::int64_t> InternalDisassembler::determine_next(std::uint64_t curr_idx) {
    return determine_next(last_instr, curr_idx);
}

std::int16_t InternalDisassembler::get_conditional_jump_target(Instruction *instr) {
    auto op_code = instr->get_instruction_opcode();

    switch (op_code) {
        case disassembler::opcodes::OP_IF_EQ:
        case disassembler::opcodes::OP_IF_NE:// "if-ne"
        case disassembler::opcodes::OP_IF_LT:// "if-lt"
        case disassembler::opcodes::OP_IF_GE:// "if-ge"
        case disassembler::opcodes::OP_IF_GT:// "if-gt"
        case disassembler::opcodes::OP_IF_LE:// "if-le"
        {
            auto *i = reinterpret_cast<Instruction22t *>(instr);
            return i->getNCCCC();
        }
        case disassembler::opcodes::OP_IF_EQZ:// "if-eqz"
        case disassembler::opcodes::OP_IF_NEZ:// "if-nez"
        case disassembler::opcodes::OP_IF_LTZ:// "if-ltz"
        case disassembler::opcodes::OP_IF_GEZ:// "if-gez"
        case disassembler::opcodes::OP_IF_GTZ:// "if-gtz"
        case disassembler::opcodes::OP_IF_LEZ:// "if-lez"
        {
            auto *i = reinterpret_cast<Instruction21t *>(instr);
            return i->getNBBBB();
        }
        default:
            return 0;
    }
}

std::int32_t InternalDisassembler::get_unconditional_jump_target(Instruction *instr) {
    auto op_code = instr->get_instruction_opcode();

    switch (op_code) {
        case disassembler::opcodes::OP_GOTO: {
            auto *goto_instr = reinterpret_cast<Instruction10t *>(instr);
            return goto_instr->getNAA();
        }
        case disassembler::opcodes::OP_GOTO_16: {
            auto *goto16_instr = reinterpret_cast<Instruction20t *>(instr);
            return goto16_instr->getNAAAA();
        }
        case disassembler::opcodes::OP_GOTO_32: {
            auto *goto32_instr = reinterpret_cast<Instruction30t *>(instr);
            return goto32_instr->getNAAAAAAAA();
        }
        default:
            return 0;
    }
}

std::vector<disassembler::exception_data_t> InternalDisassembler::determine_exception(EncodedMethod *method) {
    using try_encoded = std::pair<shuriken::dex::TryItem *, shuriken::dex::EncodedCatchHandler *>;
    using vector_try_encoded = std::vector<try_encoded>;

    std::unordered_map<std::uint64_t, vector_try_encoded> h_off;

    disassembler::exceptions_data_t exceptions;

    if (!method || !method->get_code_items()->get_number_try_items())
        return {};

    auto *code_item = method->get_code_items();

    // retrieve all the try items with the handler
    // of the offset
    for (auto &try_item: code_item->get_try_items()) {
        auto offset_handler = try_item.handler_off +
                              code_item->get_encoded_catch_handler_list_offset();
        h_off[offset_handler].push_back({&try_item, nullptr});
    }

    // add the encoded catch handlers to the structure
    for (auto &encoded_catch_handler: code_item->get_encoded_catch_handlers()) {
        auto it = h_off.find(encoded_catch_handler.get_offset());

        if (it == h_off.end())
            continue;

        for (auto &v: it->second)
            v.second = &encoded_catch_handler;
    }

    // now create the exceptions structure
    for (auto &off_values: h_off) {
        for (auto &values: off_values.second) {
            auto *try_value = values.first;
            auto *handler_catch = values.second;

            disassembler::exception_data_t z;

            z.try_value_start_addr = try_value->start_addr * 2;
            z.try_value_end_addr = (try_value->start_addr * 2) +
                                   (try_value->insn_count * 2);

            for (auto &catch_type_pair: handler_catch->get_handle_pairs())
                z.handler.push_back({catch_type_pair.type, catch_type_pair.idx * 2});

            exceptions.emplace_back(z);
        }
    }

    return exceptions;
}

void InternalDisassembler::assign_switch_if_any(std::list<std::unique_ptr<Instruction>> &instructions,
                                                std::unordered_map<std::uint64_t, Instruction *> &cache_instructions) {
    for (auto &instr: instructions) {
        auto op_code = instr->get_instruction_opcode();

        if (op_code == disassembler::opcodes::OP_PACKED_SWITCH ||
            op_code == disassembler::opcodes::OP_SPARSE_SWITCH) {
            auto *instr31t = reinterpret_cast<Instruction31t *>(instr.get());

            auto switch_idx = instr31t->get_address() + (instr31t->getNBBBBBBBB() * 2);

            auto it = cache_instructions.find(switch_idx);

            if (it != cache_instructions.end()) {
                if (op_code == disassembler::opcodes::OP_PACKED_SWITCH)
                    instr31t->set_packed_switch(reinterpret_cast<PackedSwitch *>(it->second));
                else// DexOpcodes::opcodes::OP_SPARSE_SWITCH
                    instr31t->set_sparse_switch(reinterpret_cast<SparseSwitch *>(it->second));
            }
        }
    }
}

std::list<std::unique_ptr<Instruction>>
InternalDisassembler::disassemble(std::span<std::uint8_t> buffer_bytes) {
    std::unordered_map<std::uint64_t, Instruction *> cache_instr;
    std::uint64_t idx = 0;                                          // index of the instr
    std::list<std::unique_ptr<Instruction>> instructions;           // all the instructions from the method
    std::unique_ptr<Instruction> instr;                             // instruction to create
    auto buffer_size = buffer_bytes.size();                  // size of the buffer
    disassembler::opcodes opcode;                                   // opcode of the operations
    bool exist_switch = false;                                      // check a switch exist

    while (idx < buffer_size) {
        opcode = static_cast<disassembler::opcodes>(buffer_bytes[idx]);

        if (!exist_switch &&
            (opcode == disassembler::opcodes::OP_PACKED_SWITCH
             || opcode == disassembler::opcodes::OP_SPARSE_SWITCH))
            exist_switch = true;

        instr = disassemble_instruction(opcode,
                                        buffer_bytes,
                                        idx);
        if (instr) {
            if (!instr->is_instruction_valid()) {
                DalvikIncorrectInstruction::Impl *impl = new DalvikIncorrectInstruction::Impl(buffer_bytes,
                                                                                              idx,
                                                                                              instr->get_error_message(),
                                                                                              instr->get_instruction_length(),
                                                                                              instr->get_address(),
                                                                                              instr->get_instruction_opcode());
                instr = std::make_unique<DalvikIncorrectInstruction>(impl);
            }
            instr->set_address(idx);
            instructions.push_back(std::move(instr));
            cache_instr[idx] = instructions.back().get();
            idx += instructions.back()->get_instruction_length();
        } else {
            idx += 1;
        }
    }

    if (exist_switch)
        assign_switch_if_any(instructions, cache_instr);

    return instructions;
}
