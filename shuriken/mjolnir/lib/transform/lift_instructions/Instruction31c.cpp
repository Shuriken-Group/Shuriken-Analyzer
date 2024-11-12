
#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction31c *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code) {
        case DexOpcodes::opcodes::OP_CONST_STRING_JUMBO: {
            auto str_value = instr->get_string_value();
            auto str_ref = instr->get_string_idx();

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::LoadString>(
                    location,
                    strObjectType,
                    str_value,
                    str_ref);

            writeVariable(current_basic_block, dest, gen_value);
        } break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction31c not supported");
            break;
    }
}
