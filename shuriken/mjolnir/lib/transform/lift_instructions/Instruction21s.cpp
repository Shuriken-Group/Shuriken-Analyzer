
#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction21s *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    mlir::Type dest_type;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_CONST_16:
            if (!dest_type)
                dest_type = intType;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_CONST_WIDE_16:
            if (!dest_type)
                dest_type = longType;
            {
                auto value = static_cast<std::int64_t>(instr->get_source());

                auto gen_value = builder.create<::mlir::shuriken::MjolnIR::LoadValue>(
                        location,
                        dest_type,
                        value);
                writeVariable(current_basic_block, dest, gen_value);
            }
            break;
        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction21s not supported");
            break;
    }
}
