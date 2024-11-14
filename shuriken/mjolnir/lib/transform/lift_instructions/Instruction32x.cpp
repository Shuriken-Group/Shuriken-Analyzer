


#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction32x *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    switch (op_code) {
        case DexOpcodes::opcodes::OP_MOVE_16:
        case DexOpcodes::opcodes::OP_MOVE_WIDE_16:
        case DexOpcodes::opcodes::OP_MOVE_OBJECT_16: {
            auto src_value = readVariable(current_basic_block, current_method->get_basic_blocks(), src);

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::MoveOp>(
                    location,
                    src_value.getType(),
                    src_value);

            writeVariable(current_basic_block, dest, gen_value);
        } break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction32x not supported");
            break;
    }
}
