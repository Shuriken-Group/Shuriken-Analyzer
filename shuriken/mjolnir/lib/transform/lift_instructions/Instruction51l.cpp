

#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction51l *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    /// we will take the registers as big enough
    /// for storing a 64-bit value
    auto dest_reg = instr->get_first_register();

    mlir::Type dest_type;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_CONST_WIDE:
            if (!dest_type)
                dest_type = doubleType;
            {
                auto value = static_cast<std::int64_t>(instr->get_wide_value());

                auto gen_value = builder.create<::mlir::shuriken::MjolnIR::LoadValue>(
                        location,
                        dest_type,
                        value);

                writeLocalVariable(current_basic_block, dest_reg, gen_value);
            }
            break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction51l not supported");
            break;
    }
}
