
#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction21t *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto v1 = instr->get_check_reg();

    auto v2 = builder.create<::mlir::arith::ConstantIntOp>(location, 0, 32);

    mlir::Value cmp_value;

    mlir::Type I1 = ::mlir::IntegerType::get(&context, 1);

    switch (op_code) {
        case DexOpcodes::opcodes::OP_IF_EQZ:
            if (!cmp_value) {
                cmp_value = builder.create<::mlir::shuriken::MjolnIR::CmpEqz>(
                        location,
                        I1,
                        readVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                        v2);
            }
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_IF_NEZ:
            if (!cmp_value) {
                cmp_value = builder.create<::mlir::shuriken::MjolnIR::CmpNeqz>(
                        location,
                        I1,
                        readVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                        v2);
            }
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_IF_LTZ:
            if (!cmp_value) {
                cmp_value = builder.create<::mlir::shuriken::MjolnIR::CmpLtz>(
                        location,
                        I1,
                        readVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                        v2);
            }
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_IF_GEZ:
            if (!cmp_value) {
                cmp_value = builder.create<::mlir::shuriken::MjolnIR::CmpGez>(
                        location,
                        I1,
                        readVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                        v2);
            }
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_IF_GTZ:
            if (!cmp_value) {
                cmp_value = builder.create<::mlir::shuriken::MjolnIR::CmpGtz>(
                        location,
                        I1,
                        readVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                        v2);
            }
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_IF_LEZ: {
            if (!cmp_value) {
                cmp_value = builder.create<::mlir::shuriken::MjolnIR::CmpLez>(
                        location,
                        I1,
                        readVariable(current_basic_block, current_method->get_basic_blocks(), v1),
                        v2);
            }

            auto location_jcc = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 1);
            /// get the addresses from the blocks
            auto true_idx = instr->get_address() + (instr->get_jump_offset() * 2);
            auto false_idx = instr->get_address() + instr->get_instruction_length();
            /// get the blocks:
            ///     - current_block: for obtaining the required arguments.
            ///     - true_block: for generating branch to `true` block
            ///     - false_block: for generating fallthrough to `false` block.
            auto true_block = current_method->get_basic_blocks()->get_basic_block_by_idx(true_idx);
            auto false_block = current_method->get_basic_blocks()->get_basic_block_by_idx(false_idx);
            /// create the conditional branch
            builder.create<::mlir::cf::CondBranchOp>(
                    location_jcc,
                    cmp_value,
                    map_blocks[true_block],
                    CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, true_block)],
                    map_blocks[false_block],
                    CurrentDef[current_basic_block].jmpParameters[std::make_pair(current_basic_block, false_block)]);
        }

        break;
        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Error Instruction21t not supported");
            break;
    }
}
