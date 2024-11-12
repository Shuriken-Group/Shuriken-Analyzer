

#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction31i *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    mlir::Type dest_type;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_CONST:
        case DexOpcodes::opcodes::OP_CONST_WIDE_32: {
            /// for the moment set destination type as a long,
            /// we need to think a better algorithm
            if (!dest_type)
                dest_type = longType;

            auto value = static_cast<std::int64_t>(instr->get_source());

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::LoadValue>(
                    location,
                    dest_type,
                    value);

            writeVariable(current_basic_block, dest, gen_value);
        } break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction31i not supported");
            break;
    }
}
