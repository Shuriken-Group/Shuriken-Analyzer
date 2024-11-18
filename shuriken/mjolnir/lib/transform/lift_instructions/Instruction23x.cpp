#include "mjolnir/MjolnIREnums.h"
#include "transform/lifter.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/OpDefinition.h>
#include <optional>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction23x *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src1 = instr->get_first_source();
    auto src2 = instr->get_second_source();

    mlir::Type dest_type = nullptr;
    std::optional<WidthEnum> width = std::nullopt;

    switch (op_code) {
        /// Different Add Operations
        case DexOpcodes::opcodes::OP_ADD_INT:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_ADD_LONG:
            if (!dest_type)
                dest_type = longType;
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<mlir::arith::AddIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
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

                auto generated_value = builder.create<mlir::arith::AddFOp>(
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
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<mlir::arith::SubIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
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

                auto generated_value = builder.create<mlir::arith::SubFOp>(
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
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<mlir::arith::MulIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
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

                auto generated_value = builder.create<mlir::arith::MulFOp>(
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
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<mlir::arith::DivSIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
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

                auto generated_value = builder.create<mlir::arith::DivFOp>(
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
            {
                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);

                auto generated_value = builder.create<::mlir::arith::RemSIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
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

                auto generated_value = builder.create<mlir::arith::RemFOp>(
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

                auto generated_value = builder.create<mlir::arith::AndIOp>(
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

                auto generated_value = builder.create<mlir::arith::OrIOp>(
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

                auto generated_value = builder.create<mlir::arith::XOrIOp>(
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

                auto generated_value = builder.create<mlir::arith::ShLIOp>(
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

                auto generated_value = builder.create<mlir::arith::ShRSIOp>(
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

                auto generated_value = builder.create<mlir::arith::ShRUIOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value);

                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
            // 44: aget
            // 45: aget-wide
            // 46: aget-object
            // 47: aget-boolean
            // 48: aget-byte
            // 49: aget-char
            // 4a: aget-short
            // 4b: aput
            // 4c: aput-wide
            // 4d: aput-object
            // 4e: aput-boolean
            // 4f: aput-byte
            // 50: aput-char
            // 51: aput-short
        case DexOpcodes::opcodes::OP_AGET:
            if (!width) width = WidthEnum::DEFAULT;
            if (!dest_type) dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AGET_WIDE:
            if (!width) width = WidthEnum::WIDE;
            if (!dest_type) dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AGET_OBJECT:
            if (!width) width = WidthEnum::OBJECT;
            if (!dest_type) dest_type = ::mlir::shuriken::MjolnIR::DVMObjectType::get(&context, "aget-object");
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AGET_BOOLEAN:
            if (!width) width = WidthEnum::BOOLEAN;
            if (!dest_type) dest_type = boolType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AGET_BYTE:
            if (!width) width = WidthEnum::BYTE;
            if (!dest_type) dest_type = byteType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AGET_CHAR:
            if (!width) width = WidthEnum::CHAR;
            if (!dest_type) dest_type = charType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_AGET_SHORT:
            if (!width) width = WidthEnum::SHORT;
            if (!dest_type) dest_type = shortType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_APUT:
            if (!width) width = WidthEnum::DEFAULT;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_APUT_WIDE:
            if (!width) width = WidthEnum::WIDE;
            if (!dest_type) dest_type = longType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_APUT_OBJECT:
            if (!width) width = WidthEnum::OBJECT;
            if (!dest_type) dest_type = ::mlir::shuriken::MjolnIR::DVMObjectType::get(&context, "aget-object");
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_APUT_BYTE:
            if (!width) width = WidthEnum::BYTE;
            if (!dest_type) dest_type = byteType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_APUT_CHAR:
            if (!width) width = WidthEnum::CHAR;
            if (!dest_type) dest_type = charType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_APUT_SHORT:
            if (!width) width = WidthEnum::SHORT;
            if (!dest_type) dest_type = shortType;
            {

                auto src1_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src1);
                auto src2_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src2);
                auto generated_value = builder.create<mlir::shuriken::MjolnIR::GetArrayOp>(
                        location,
                        dest_type,
                        src1_value,
                        src2_value, *width);
                writeVariable(current_basic_block, dest, generated_value);
            }
            break;
        default:
            throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Opcode from Instruction23x not implemented");
            break;
    }
}
