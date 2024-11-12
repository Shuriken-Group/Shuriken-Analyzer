

#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction11n *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code) {
        case DexOpcodes::opcodes::OP_CONST_4: {
            auto value = instr->get_source();

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::LoadValue>(
                    location,
                    ::mlir::shuriken::MjolnIR::DVMByteType::get(&context),
                    value);

            writeVariable(current_basic_block, dest, gen_value);
        } break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction11n not supported");
            break;
    }
}
