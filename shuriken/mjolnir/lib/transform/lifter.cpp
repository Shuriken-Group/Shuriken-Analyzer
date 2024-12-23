
#include "transform/lifter.h"
#include "mjolnir/MjolnIRDialect.h"
#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"
#include "shuriken/analysis/Dex/dex_analysis.h"
#include "shuriken/common/Dex/dvm_types.h"
#include "shuriken/common/logger.h"
#include "shuriken/disassembler/Dex/dex_instructions.h"
#include "shuriken/disassembler/Dex/dex_opcodes.h"
#include "shuriken/disassembler/Dex/internal_disassembler.h"
#include "shuriken/exceptions/invalidinstruction_exception.h"
#include "shuriken/parser/Dex/dex_protos.h"
#include "shuriken/parser/Dex/dex_types.h"

/// MLIR includes
#include <cassert>
#include <llvm/ADT/StringExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

#include <iterator>
#include <mlir/IR/OwningOpRef.h>
#include <string>
#include <utility>


using namespace shuriken::MjolnIR;
using shuriken::analysis::dex::BasicBlocks;
using shuriken::analysis::dex::DVMBasicBlock;
using shuriken::analysis::dex::MethodAnalysis;
using namespace shuriken::disassembler::dex;
using shuriken::disassembler::dex::InstructionUtils;
using shuriken::parser::dex::ARRAY;
using shuriken::parser::dex::CLASS;
using shuriken::parser::dex::DVMClass;
using shuriken::parser::dex::DVMFundamental;
using shuriken::parser::dex::DVMType;
using shuriken::parser::dex::FUNDAMENTAL;
using shuriken::parser::dex::fundamental_e;
using shuriken::parser::dex::ProtoID;
Lifter::Lifter(const std::string &file_name, bool gen_exception, bool LOGGING)
    : context(mlir::MLIRContext()), builder(&context), gen_exception(gen_exception), LOGGING(LOGGING) {
    parser = shuriken::parser::parse_dex(file_name);
    assert(parser);
    disassembler = std::make_unique<shuriken::disassembler::dex::DexDisassembler>(parser.get());
    assert(disassembler);
    disassembler->disassembly_dex();

    // INFO: xrefs option disabled
    analysis = std::make_unique<shuriken::analysis::dex::Analysis>(parser.get(), disassembler.get(), false);
    auto mm = analysis->get_methods();
    init();
    mlir_gen_result = mlirGen();
}
void Lifter::init() {
    registry.insert<::mlir::shuriken::MjolnIR::MjolnIRDialect>();
    context.loadAllAvailableDialects();
    context.getOrLoadDialect<::mlir::shuriken::MjolnIR::MjolnIRDialect>();
    context.getOrLoadDialect<::mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<::mlir::arith::ArithDialect>();
    context.getOrLoadDialect<::mlir::func::FuncDialect>();

    voidType = ::mlir::shuriken::MjolnIR::DVMVoidType::get(&context);
    byteType = ::mlir::IntegerType::get(&context, 8, mlir::IntegerType::Signed);// signed at default
    boolType = ::mlir::IntegerType::get(&context, 1, mlir::IntegerType::Signless);
    charType = ::mlir::IntegerType::get(&context, 8, mlir::IntegerType::Signed);
    shortType = ::mlir::IntegerType::get(&context, 16, mlir::IntegerType::Signed);
    intType = builder.getI32Type();// INFO: AddIOp can only take I32, not SI or UI32
    longType = ::mlir::IntegerType::get(&context, 64, mlir::IntegerType::Signed);
    floatType = ::mlir::Float32Type::get(&context);
    doubleType = ::mlir::Float64Type::get(&context);
    strObjectType = ::mlir::shuriken::MjolnIR::DVMObjectType::get(&context, "Ljava/lang/String;");
}

mlir::Type Lifter::get_type(DVMFundamental *fundamental) {
    switch (fundamental->get_fundamental_type()) {
        case fundamental_e::BOOLEAN:
            return boolType;
        case fundamental_e::BYTE:
            return byteType;
        case fundamental_e::CHAR:
            return charType;
        case fundamental_e::DOUBLE:
            return doubleType;
        case fundamental_e::FLOAT:
            return floatType;
        case fundamental_e::INT:
            return intType;
        case fundamental_e::LONG:
            return longType;
        case fundamental_e::SHORT:
            return shortType;
        default:
            return voidType;
    }
}

mlir::Type Lifter::get_type(DVMClass *cls) {
    return ::mlir::shuriken::MjolnIR::DVMObjectType::get(&context, cls->get_class_name());
}

mlir::Type Lifter::get_type(DVMType *type) {
    if (type->get_type() == FUNDAMENTAL)
        return get_type(reinterpret_cast<DVMFundamental *>(type));
    else if (type->get_type() == CLASS)
        return get_type(reinterpret_cast<DVMClass *>(type));
    else if (type->get_type() == ARRAY) {
        DVMArray *dvm_array_p = reinterpret_cast<DVMArray *>(type);
        return ::mlir::shuriken::MjolnIR::DVMArrayType::get(&context, dvm_array_p->print_type());
        // throw exceptions::LifterException("MjolnIRLIfter::get_type: type ARRAY not implemented yet...");
    } else

        throw exceptions::LifterException("MjolnIRLifter::get_type: that type is unknown or I don't know what it is...");
}

llvm::SmallVector<mlir::Type> Lifter::gen_prototype(ProtoID *proto, bool is_static, DVMType *cls) {
    llvm::SmallVector<mlir::Type, 4> argTypes;

    /// as much space as parameters
    // argTypes.reserve(proto->get_parameters().size());

    if (!is_static)
        argTypes.push_back(get_type(cls));

    /// since we have a vector of parameters
    /// it is easy peasy
    for (auto param: proto->get_parameters())
        argTypes.push_back(get_type(param));

    return argTypes;
}

::mlir::shuriken::MjolnIR::MethodOp Lifter::get_method(analysis::dex::MethodAnalysis *M) {
    auto encoded_method = M->get_encoded_method();

    auto method = encoded_method->getMethodID();

    parser::dex::ProtoID *proto = method->get_prototype();
    auto name = method->get_method_name();

    auto method_location = mlir::FileLineColLoc::get(&context, llvm::StringRef(name), 0, 0);

    // now let's create a MethodOp, for that we will need first to retrieve
    // the type of the parameters
    bool is_static = (encoded_method->get_flags() & shuriken::dex::TYPES::access_flags::ACC_STATIC) == shuriken::dex::TYPES::access_flags::ACC_STATIC;
    auto paramTypes = gen_prototype(proto, is_static, method->get_class());

    // now retrieve the return type
    mlir::Type retType = get_type(proto->get_return_type());

    // create now the method type
    auto methodType = builder.getFunctionType(paramTypes, {retType});

    auto methodOp = builder.create<::mlir::shuriken::MjolnIR::MethodOp>(method_location, name, methodType);

    /// declare the register parameters, these are used during the
    /// program
    auto number_of_params = std::ranges::distance(proto->get_parameters());
    if (!is_static) number_of_params += 1;

    auto number_of_registers = encoded_method->get_code_item()->get_registers_size();

    auto first_block = M->get_basic_blocks()->get_basic_block_by_idx(0);

    for (std::uint32_t Reg = (number_of_registers - number_of_params),/// starting index of the parameter
         Limit = (static_cast<std::uint32_t>(number_of_registers)),   /// limit value for parameters
         Argument = 0;                                                /// for obtaining parameter by index 0
         Reg < Limit;
         ++Reg,
                       ++Argument) {
        /// get the value from the parameter
        auto value = methodOp.getArgument(Argument);
        /// write to a local variable
        writeVariable(first_block, Reg, value);
    }

    // with the type created, now create the Method
    return methodOp;
}


/// INFO: Algorithm from Braun et al.
/// If we cant read the variable's ssa value from our own block (readVariable)
///   then we recursively read our own block's predecessors.
mlir::Value Lifter::readVariableRecursive(analysis::dex::DVMBasicBlock *BB,
                                          analysis::dex::BasicBlocks *BBs,
                                          std::uint32_t Reg) {
    mlir::Value new_value;

    /// because block doesn't have it add it to required.
    CurrentDef[BB].required.insert(Reg);
    for (auto pred: BBs->predecessors(BB)) {
        if (!CurrentDef[pred].Filled)
            gen_block(pred);
        auto Val = readVariable(pred, BBs, Reg);

        /// if the value is required, add the argument to the block
        /// write the local variable and erase from required
        if (CurrentDef[BB].required.find(Reg) != CurrentDef[BB].required.end()) {
            auto Loc = mlir::FileLineColLoc::get(&context, module_name, BB->get_first_address(), 0);

            new_value = map_blocks[BB]->addArgument(Val.getType(), Loc);

            writeVariable(BB, Reg, new_value);

            CurrentDef[BB].required.erase(Reg);
        }

        CurrentDef[pred].jmpParameters[{pred, BB}].push_back(Val);
    }

    return new_value;
}
/// INFO: First level in the lifting process, we create a MethodOp (with get_method()) as well as
///   map out all of DVMBasicBlock to their respective Block.
void Lifter::gen_method(MethodAnalysis *method) {
    /// create the method
    auto function = get_method(method);
    /// obtain the basic blocks
    auto bbs = method->get_basic_blocks();
    /// update the current method
    current_method = method;

    /// generate the blocks for each node
    auto bb_nodes = bbs->nodes();
    for (auto it = bb_nodes.begin(); it != bb_nodes.end(); it++) {
        auto bb = *it;
        // TODO: Need Edu for code review
        // if (it == bb_nodes.begin() || it == std::prev(bb_nodes.end()))
        //     continue;
        if (bb->get_first_address() == 0)// if it's the first block
        {
            auto &entryBlock = function.front();
            map_blocks[bb] = &entryBlock;
        } else// others must be generated
            map_blocks[bb] = function.addBlock();
    }
    /// now traverse each node for generating instructions
    for (auto it = bb_nodes.begin(); it != bb_nodes.end(); it++) {
        auto bb = *it;
        // TODO: Need Edu for code review
        // if (it == bb_nodes.begin() || it == std::prev(bb_nodes.end()))
        //     continue;
        /// set as the insertion point of the instructions
        builder.setInsertionPointToStart(map_blocks[bb]);

        gen_block(bb);
    }

    for (auto bb: bb_nodes)
        gen_terminators(bb);
}

/// INFO: Generate an mlir basic block full with the transform instruction
/// Setting current_basic_block to bb allows the builder in gen_instruction to insert the
///   transformed instruction to the correct basic block.
/// Setting CurrentDef[bb].Filled to 1 since we've filled the block (except for its terminator)
///
/// INFO: Filling the block (CurrentDef[bb].Filled) is a necessary condition for readVariableRecursive on a block
void Lifter::gen_block(analysis::dex::DVMBasicBlock *bb) {
    /// update current basic block
    // this->log(fmt::format("Gen_Block of {}", bb->toString()));
    current_basic_block = bb;

    for (auto instr: bb->get_instructions()) {
        try {
            auto operation = InstructionUtils::get_operation_type_from_opcode(static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode()));
            /// we will generate terminators later
            if (instr->is_terminator() &&
                operation != shuriken::disassembler::dex::DexOpcodes::RET_BRANCH_DVM_OPCODE)
                continue;
            /// generate the instruction
            gen_instruction(instr);
        } catch (const exceptions::LifterException &e) {
            /// if user wants to generate exception
            if (gen_exception)
                throw e;
            /// if not just create a Nop instruction
            auto Loc = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);
            builder.create<::mlir::shuriken::MjolnIR::Nop>(Loc, e.what());
        }
    }

    CurrentDef[bb].Filled = 1;
}

/// INFO: Function to add terminator to a block,
/// If the last instruction is naturally a terminator, we generate it like normal
/// Otherwise, artifically create a FallThrough Op to differentiate itself from a Cond.Br
/// *** MLIR requires each block to end with a terminator
void Lifter::gen_terminators(DVMBasicBlock *bb) {
    current_basic_block = bb;

    auto last_instr = bb->get_instructions().back();

    builder.setInsertionPointToEnd(map_blocks[bb]);
    try {
        auto operation = shuriken::disassembler::dex::InstructionUtils::get_operation_type_from_opcode(static_cast<disassembler::dex::DexOpcodes::opcodes>(last_instr->get_instruction_opcode()));

        if (operation == disassembler::dex::DexOpcodes::RET_BRANCH_DVM_OPCODE)
            return;
        if (last_instr->is_terminator())
            gen_instruction(last_instr);
        else {

            auto next_block = current_method->get_basic_blocks()->get_basic_block_by_idx(
                    last_instr->get_address() + last_instr->get_instruction_length());
            auto loc = mlir::FileLineColLoc::get(&context, module_name, last_instr->get_address(), 1);
            builder.create<::mlir::shuriken::MjolnIR::FallthroughOp>(
                    loc,
                    map_blocks[next_block],
                    CurrentDef[bb].jmpParameters[{bb, next_block}]);
        }
    } catch (const exceptions::LifterException &e) {
        /// if user wants to generate exception
        if (gen_exception)
            throw e;
        /// if not just create a Nop instruction
        auto Loc = mlir::FileLineColLoc::get(&context, module_name, last_instr->get_address(), 0);
        builder.create<::mlir::shuriken::MjolnIR::Nop>(Loc, e.what());
    }
}

/// INFO: Entry point to calling the lifter
///   It sets the module name to the method name for lowering purposes later, then calls gen_method
std::vector<mlir::OwningOpRef<mlir::ModuleOp>> Lifter::mlirGen() {
    auto cc = analysis->get_classes();

    /// Create a Module per class
    /// modules will contain a MethodOp for each method
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> result;

    for (auto &[class_name, class_ref]: cc) {
        auto Module = mlir::ModuleOp::create(builder.getUnknownLoc());

        Module.setName(class_name);

        for (auto &[method_name, method_analysis]: class_ref.get().get_methods()) {
            builder.setInsertionPointToEnd(Module.getBody());

            gen_method(method_analysis);
        }

        result.push_back(Module);
    }

    return result;
}

/// INFO: Entry point to each instruction transform from DEI to Mjolnir
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction *instr) {
    using namespace shuriken::disassembler::dex;
    using shuriken::disassembler::dex::Instruction;
    switch (instr->get_instruction_type()) {
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION23X:
            gen_instruction(reinterpret_cast<Instruction23x *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION12X:
            gen_instruction(reinterpret_cast<Instruction12x *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION11X:
            gen_instruction(reinterpret_cast<Instruction11x *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION22C:
            gen_instruction(reinterpret_cast<Instruction22c *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION22T:
            gen_instruction(reinterpret_cast<Instruction22t *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION21T:
            gen_instruction(reinterpret_cast<Instruction21t *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION10T:
            gen_instruction(reinterpret_cast<Instruction10t *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION20T:
            gen_instruction(reinterpret_cast<Instruction20t *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION30T:
            gen_instruction(reinterpret_cast<Instruction30t *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION10X:
            gen_instruction(reinterpret_cast<Instruction10x *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION11N:
            gen_instruction(reinterpret_cast<Instruction11n *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION21S:
            gen_instruction(reinterpret_cast<Instruction21s *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION21H:
            gen_instruction(reinterpret_cast<Instruction21h *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION51L:
            gen_instruction(reinterpret_cast<Instruction51l *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION35C:
            gen_instruction(reinterpret_cast<Instruction35c *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION21C:
            gen_instruction(reinterpret_cast<Instruction21c *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION22X:
            gen_instruction(reinterpret_cast<Instruction22x *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION32X:
            gen_instruction(reinterpret_cast<Instruction32x *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION31I:
            gen_instruction(reinterpret_cast<Instruction31i *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION31C:
            gen_instruction(reinterpret_cast<Instruction31c *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION22S:
            gen_instruction(reinterpret_cast<Instruction22s *>(instr));
            break;
        case DexOpcodes::dexinsttype::DEX_INSTRUCTION22B:
            gen_instruction(reinterpret_cast<Instruction22b *>(instr));
            break;
        default:
            throw exceptions::LifterException("MjolnIRLifter::gen_instruction: InstructionType not implemented");
    }
}
BasicBlockType Lifter::get_block_type(::mlir::Block *bb) {
    return this->block_type_map[bb];
}
BasicBlockType Lifter::get_block_type(DVMBasicBlock *bb) {
    return this->block_type_map[this->map_blocks[bb]];
}
void Lifter::set_block_type(::mlir::Block *bb, BasicBlockType bt) {
    this->block_type_map[bb] = bt;
}
void Lifter::set_block_type(DVMBasicBlock *bb, BasicBlockType bt) {
    this->block_type_map[this->map_blocks[bb]] = bt;
}
