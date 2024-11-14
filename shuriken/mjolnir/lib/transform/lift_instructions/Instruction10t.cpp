
#include "shuriken/common/Dex/dvm_types.h"
#include "transform/lifter.h"
#include <cstdint>
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction10t *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());
    // TODO: is all instruction per line only? as in 1 instruction per line
    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    this->log(fmt::format("Gen instr {}", instr->print_instruction()));

    switch (op_code) {
        case DexOpcodes::opcodes::OP_GOTO: {
            auto offset = instr->get_offset();
            auto target_idx = instr->get_address() + (offset * 2);

            auto target_block = current_method->get_basic_blocks()->get_basic_block_by_idx(target_idx);

            builder.create<::mlir::cf::BranchOp>(
                    location,
                    map_blocks[target_block],
                    CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, target_block)]);
        } break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction10t not supported or not recognized.");
            break;
    }
}
