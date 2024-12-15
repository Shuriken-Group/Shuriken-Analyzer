

#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>
#include <string_view>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction22c *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto regA = instr->get_destination();
    auto regB = instr->get_operand();

    mlir::Type destination_type;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_IGET:
        case DexOpcodes::opcodes::OP_IGET_WIDE:
        case DexOpcodes::opcodes::OP_IGET_OBJECT:
        case DexOpcodes::opcodes::OP_IGET_BOOLEAN:
        case DexOpcodes::opcodes::OP_IGET_BYTE:
        case DexOpcodes::opcodes::OP_IGET_CHAR:
        case DexOpcodes::opcodes::OP_IGET_SHORT: {
            auto field = std::get<FieldID *>(instr->get_checked_id_as_kind());
            auto regB_value = readVariable(current_basic_block, current_method->get_basic_blocks(), regB);

            std::string_view field_name = field->field_name();
            std::string_view field_class = field->field_class()->get_raw_type();

            if (!destination_type)
                destination_type = get_type(field->field_type());

            auto generated_value = builder.create<::mlir::shuriken::MjolnIR::LoadFieldOp>(
                    location,
                    field_name,
                    field_class,
                    regB_value,
                    destination_type);

            writeVariable(current_basic_block, regA, generated_value);
        } break;
        case DexOpcodes::opcodes::OP_IPUT:
        case DexOpcodes::opcodes::OP_IPUT_WIDE:
        case DexOpcodes::opcodes::OP_IPUT_OBJECT:
        case DexOpcodes::opcodes::OP_IPUT_BOOLEAN:
        case DexOpcodes::opcodes::OP_IPUT_BYTE:
        case DexOpcodes::opcodes::OP_IPUT_CHAR:
        case DexOpcodes::opcodes::OP_IPUT_SHORT: {
            auto field = std::get<FieldID *>(instr->get_checked_id_as_kind());
            auto regA_value = readVariable(current_basic_block, current_method->get_basic_blocks(), regA);
            auto regB_value = readVariable(current_basic_block, current_method->get_basic_blocks(), regB);

            std::string_view field_name = field->field_name();
            std::string_view field_class = field->field_class()->get_raw_type();

            builder.create<::mlir::shuriken::MjolnIR::StoreFieldOp>(
                    location,
                    regA_value,
                    field_name,
                    field_class,
                    regB_value);
        } break;
        default:
            throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction22c not implemented yet");
            break;
    }
}
