
#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction22b *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);
    auto location_1 = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 1);

    auto dest = instr->get_destination();
    auto src1 = instr->get_first_operand();
    auto src2 = instr->get_second_operand();

    mlir::Value val;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_ADD_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::AddOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_SUB_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::SubOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_MUL_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::MulOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_DIV_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::DivOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_REM_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::RemOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_AND_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::AndOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_OR_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::OrOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_XOR_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::XorOp>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_SHL_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Shl>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_SHR_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Shr>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_USHR_INT_LIT8:
            if (!val)
                val = builder.create<mlir::arith::ConstantIntOp>(location, src2, 8);
            {
                auto src1_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), src1);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Shr>(
                        location_1,
                        intType,
                        src1_value,
                        val);

                writeLocalVariable(current_basic_block, dest, generated_value);
            }
            break;
        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction22b not supported");
            break;
    }
}
