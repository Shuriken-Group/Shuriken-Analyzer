

#include "shuriken/disassembler/Dex/dex_opcodes.h"
#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <memory>
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction11x *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code) {
        case DexOpcodes::opcodes::OP_RETURN:
        case DexOpcodes::opcodes::OP_RETURN_WIDE:
        case DexOpcodes::opcodes::OP_RETURN_OBJECT: {
            auto reg_value = readVariable(current_basic_block, current_method->get_basic_blocks(), dest);

            builder.create<::mlir::shuriken::MjolnIR::ReturnOp>(
                    location,
                    reg_value);
        } break;
        case DexOpcodes::opcodes::OP_MOVE_RESULT:
        case DexOpcodes::opcodes::OP_MOVE_RESULT_WIDE:
        case DexOpcodes::opcodes::OP_MOVE_RESULT_OBJECT: {
            if (auto call = mlir::dyn_cast<::mlir::shuriken::MjolnIR::InvokeOp>(map_blocks[current_basic_block]->back())) {
                if (call.getNumResults() == 0)
                    break;
                auto call_result = call.getResult(0);
                writeVariable(current_basic_block, dest, call_result);
            } else
                throw exceptions::LifterException("Lifter::gen_instruction: error lifting OP_MOVE_RESULT*, last instruction is not an invoke...");
        } break;
        default:
            throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction11x not supported");
            break;
    }
}
