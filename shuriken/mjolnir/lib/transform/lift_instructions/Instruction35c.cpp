
#include "shuriken/parser/Dex/dex_methods.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction35c *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code) {
        case DexOpcodes::opcodes::OP_INVOKE_VIRTUAL:
        case DexOpcodes::opcodes::OP_INVOKE_SUPER:
        case DexOpcodes::opcodes::OP_INVOKE_DIRECT:
        case DexOpcodes::opcodes::OP_INVOKE_STATIC:
        case DexOpcodes::opcodes::OP_INVOKE_INTERFACE: {
            mlir::SmallVector<mlir::Value, 4> parameters;

            auto called_method = std::get<MethodID *>(instr->get_value());
            auto method_ref = instr->get_type_idx();
            auto method_name = called_method->get_method_name();

            auto parameters_protos = called_method->get_prototype()->get_parameters();
            auto invoke_parameters = instr->get_registers();

            ::mlir::Type retType = get_type(called_method->get_prototype()->get_return_type());

            bool is_static = op_code == DexOpcodes::opcodes::OP_INVOKE_STATIC ? true : false;

            for (size_t I = 0, P = 0, Limit = invoke_parameters.size();
                 I < Limit;
                 ++I) {
                parameters.push_back(readVariable(current_basic_block,
                                                  current_method->get_basic_blocks(), invoke_parameters[I]));

                /// If the method is not static, the first
                /// register is a pointer to the object
                if (I == 0 && op_code != DexOpcodes::opcodes::OP_INVOKE_STATIC)
                    continue;

                auto fundamental = reinterpret_cast<DVMFundamental *>(*(parameters_protos.begin() + P));

                /// if the parameter is a long or a double, skip the second register
                if (fundamental &&
                    (fundamental->get_fundamental_type() == fundamental_e::LONG ||
                     fundamental->get_fundamental_type() == fundamental_e::DOUBLE))
                    ++I;// skip next register
                /// go to next parameter
                P++;
            }

            if (mlir::isa<::mlir::shuriken::MjolnIR::DVMVoidType>(retType)) {
                mlir::Type NoneType;
                builder.create<::mlir::shuriken::MjolnIR::InvokeOp>(
                        location,
                        NoneType,
                        method_name,
                        method_ref,
                        is_static,
                        parameters);
            } else
                builder.create<::mlir::shuriken::MjolnIR::InvokeOp>(
                        location,
                        retType,
                        method_name,
                        method_ref,
                        is_static,
                        parameters);
        }
        /* code */
        break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction35c not supported");
            break;
    }
}
