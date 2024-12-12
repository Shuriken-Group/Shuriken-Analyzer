
#include "shuriken/parser/Dex/dex_methods.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>

using namespace shuriken::MjolnIR;
using namespace ::mlir::shuriken::MjolnIR;

void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction35c *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    InvokeType type = InvokeType::NONE;

    switch (op_code) {
        case DexOpcodes::opcodes::OP_INVOKE_VIRTUAL:
            if (type == InvokeType::NONE)
                type = InvokeType::VIRTUAL;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INVOKE_SUPER:
            if (type == InvokeType::NONE)
                type = InvokeType::SUPER;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INVOKE_DIRECT:
            if (type == InvokeType::NONE)
                type = InvokeType::DIRECT;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INVOKE_STATIC:
            if (type == InvokeType::NONE)
                type = InvokeType::STATIC;
            [[fallthrough]];
        case DexOpcodes::opcodes::OP_INVOKE_INTERFACE: {
            if (type == InvokeType::NONE)
                type = InvokeType::INTERFACE;


            mlir::SmallVector<mlir::Value, 4> parameters;

            auto called_method = std::get<MethodID *>(instr->get_value());
            auto method_name = called_method->get_method_name();
            auto class_name = called_method->get_class()->get_raw_type();

            auto parameters_protos = called_method->get_prototype()->get_parameters();
            auto invoke_parameters = instr->get_registers();

            ::mlir::Type retType = get_type(called_method->get_prototype()->get_return_type());

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
                        class_name,
                        type,
                        parameters);
            } else
                builder.create<::mlir::shuriken::MjolnIR::InvokeOp>(
                        location,
                        retType,
                        method_name,
                        class_name,
                        type,
                        parameters);
        }
        /* code */
        break;

        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction35c not supported");
            break;
    }
}
