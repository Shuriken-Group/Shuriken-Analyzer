
#include "mjolnir/MjolnIROps.h"
#include "transform/lifter.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction12x *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    mlir::Type dest_type = nullptr;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_MOVE:
        case DexOpcodes::opcodes::OP_MOVE_WIDE:
        case DexOpcodes::opcodes::OP_MOVE_OBJECT: {
            auto src_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::MoveOp>(
                    location,
                    src_value.getType(),
                    src_value);

            writeVariable(current_basic_block, dest, gen_value);
        } break;
        case DexOpcodes::opcodes::OP_ADD_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_ADD_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::AddIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_ADD_FLOAT_2ADDR:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_ADD_DOUBLE_2ADDR:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::AddFOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_SUB_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SUB_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::SubIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_SUB_FLOAT_2ADDR:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SUB_DOUBLE_2ADDR:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::SubFOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_MUL_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_MUL_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::MulIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_MUL_FLOAT_2ADDR:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_MUL_DOUBLE_2ADDR:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::MulFOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_DIV_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_DIV_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::DivSIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_DIV_FLOAT_2ADDR:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_DIV_DOUBLE_2ADDR:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::DivFOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_REM_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_REM_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::RemSIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_REM_FLOAT_2ADDR:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_REM_DOUBLE_2ADDR:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::RemFOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_AND_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AND_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::AndIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_OR_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_OR_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::OrIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_XOR_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_XOR_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::XOrIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_SHL_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SHL_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::ShLIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_SHR_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_SHR_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::ShRSIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_USHR_INT_2ADDR:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_USHR_LONG_2ADDR:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::ShRUIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;

        case DexOpcodes::opcodes::OP_NEG_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_NEG_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Neg>(
                        location,
                        dest_type,
                        src_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_NEG_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_NEG_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto src_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<mlir::arith::NegFOp>(
                        location,
                        dest_type,
                        src_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        case DexOpcodes::opcodes::OP_NOT_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_NOT_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::Not>(
                        location,
                        dest_type,
                        src_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        /// casts
        case DexOpcodes::opcodes::OP_INT_TO_LONG:
        case DexOpcodes::opcodes::OP_FLOAT_TO_LONG:
        case DexOpcodes::opcodes::OP_DOUBLE_TO_LONG:
            if (!dest_type)
                dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INT_TO_FLOAT:
        case DexOpcodes::opcodes::OP_LONG_TO_FLOAT:
        case DexOpcodes::opcodes::OP_DOUBLE_TO_FLOAT:
            if (!dest_type)
                dest_type = floatType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INT_TO_DOUBLE:
        case DexOpcodes::opcodes::OP_LONG_TO_DOUBLE:
        case DexOpcodes::opcodes::OP_FLOAT_TO_DOUBLE:
            if (!dest_type)
                dest_type = doubleType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_LONG_TO_INT:
        case DexOpcodes::opcodes::OP_FLOAT_TO_INT:
        case DexOpcodes::opcodes::OP_DOUBLE_TO_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INT_TO_BYTE:
            if (!dest_type)
                dest_type = byteType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INT_TO_CHAR:
            if (!dest_type)
                dest_type = charType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INT_TO_SHORT:
            if (!dest_type)
                dest_type = shortType;
            {
                auto src_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

                auto generated_value = builder.create<::mlir::shuriken::MjolnIR::CastOp>(
                        location,
                        dest_type,
                        src_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        default:
            throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction12x not supported");
    }
}
