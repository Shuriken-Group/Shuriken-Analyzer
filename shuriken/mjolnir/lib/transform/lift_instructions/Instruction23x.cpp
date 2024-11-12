#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction23x *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src1 = instr->get_first_source();
    auto src2 = instr->get_second_source();

    mlir::Type dest_type = nullptr;

    switch (op_code) {
        /// Different Add Operations
        case DexOpcodes::opcodes::OP_ADD_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_ADD_LONG:
            if (!dest_type)
                dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_ADD_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_ADD_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::AddOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        /// Different Sub operations
        case DexOpcodes::opcodes::OP_SUB_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SUB_LONG:
            if (!dest_type)
                dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SUB_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SUB_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::SubOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        /// Different Mul operations
        case DexOpcodes::opcodes::OP_MUL_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_MUL_LONG:
            if (!dest_type)
                dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_MUL_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_MUL_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::MulOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        /// Different Div operations
        case DexOpcodes::opcodes::OP_DIV_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_DIV_LONG:
            if (!dest_type)
                dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_DIV_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_DIV_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::DivOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        /// Different Rem operations
        case DexOpcodes::opcodes::OP_REM_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_REM_LONG:
            if (!dest_type)
                dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_REM_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_REM_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::RemOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        /// All And operations
        case DexOpcodes::opcodes::OP_AND_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AND_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::AndOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        /// All Or operations
        case DexOpcodes::opcodes::OP_OR_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_OR_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::OrOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        /// All Xor operations
        case DexOpcodes::opcodes::OP_XOR_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_XOR_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::XorOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        /// All SHL instructions
        case DexOpcodes::opcodes::OP_SHL_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SHL_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Shl>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        /// All SHR instructions
        case DexOpcodes::opcodes::OP_SHR_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SHR_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Shr>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        /// All USHR instructions
        case DexOpcodes::opcodes::OP_USHR_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_USHR_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::UShr>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        default:
            throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Opcode from Instruction23x not implemented");
            break;
    }
}
