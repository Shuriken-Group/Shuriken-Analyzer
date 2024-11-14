
#include "shuriken/disassembler/Dex/dex_opcodes.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction10x *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code) {

        case DexOpcodes::opcodes::OP_RETURN_VOID:
            builder.create<::mlir::shuriken::MjolnIR::ReturnOp>(
                    location);
            break;
        case DexOpcodes::opcodes::OP_NOP:
            builder.create<::mlir::shuriken::MjolnIR::Nop>(
                    location);
            break;
        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction10x not supported");
            break;
    }
}
