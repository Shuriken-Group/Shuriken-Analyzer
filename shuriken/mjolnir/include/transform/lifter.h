
#ifndef LIFTER_MJOLNIR_LIFTER_HPP
#define LIFTER_MJOLNIR_LIFTER_HPP

#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"

#include "shuriken/analysis/Dex/analysis.h"

/// MLIR includes
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"


namespace shuriken {
    namespace MjolnIR {
        class Lifter {
        public:
            using edge_t =
                    std::pair<shuriken::analysis::dex::DVMBasicBlock *, shuriken::analysis::dex::DVMBasicBlock *>;

            struct BasicBlockDef {
                /// Map a register to its definition in IR
                mlir::DenseMap<std::uint32_t, mlir::Value> Defs;

                std::set<std::uint32_t> required;

                mlir::DenseMap<edge_t, mlir::SmallVector<mlir::Value, 4>> jmpParameters;

                /// Block is sealed, means no more predecessors will be
                /// added nor analyzed.
                unsigned Analyzed : 1;

                BasicBlockDef() : Analyzed(0) {}
            };

        private:
            /// @brief A map to keep the definitions of variables, and
            /// know if a basic block is completely analyzed
            mlir::DenseMap<shuriken::analysis::dex::DVMBasicBlock *, BasicBlockDef> CurrentDef;

            /// @brief Map for the Kunai basic blocks and the
            /// mlir blocks
            std::unordered_map<shuriken::analysis::dex::DVMBasicBlock *, mlir::Block *> map_blocks;

            /// @brief Write a declaration of a local register, this will be
            /// used for local value analysis
            /// @param BB block where we find the assignment
            /// @param Reg register written
            /// @param Val
            void writeLocalVariable(shuriken::analysis::dex::DVMBasicBlock *BB, std::uint32_t Reg,
                                    mlir::Value Val) {
                assert(BB && "Basic Block does not exist");
                assert(Val && "Value does not exist");
                CurrentDef[BB].Defs[Reg] = Val;
            }

            /// @brief Read a local variable from the current basic block
            /// @param BB basic block where to retrieve the data
            /// @param BBs basic blocks to retrieve the predecessors and successors
            /// @param Reg register to retrieve its Value
            /// @return value generated from an instruction.
            mlir::Value readLocalVariable(shuriken::analysis::dex::DVMBasicBlock *BB,
                                          shuriken::analysis::dex::BasicBlocks &BBs,
                                          std::uint32_t Reg) {
                assert(BB && "Basic Block does not exist");
                auto Val = CurrentDef[BB].Defs.find(Reg);
                /// if the block has the value, return it
                if (Val != CurrentDef[BB].Defs.end())
                    return Val->second;
                /// if it doesn't have the value, it becomes required for
                /// us too
                return readLocalVariableRecursive(BB, BBs, Reg);
            }

            mlir::Value readLocalVariableRecursive(shuriken::analysis::dex::DVMBasicBlock *BB,
                                                   shuriken::analysis::dex::BasicBlocks &BBs,
                                                   std::uint32_t Reg);

            /// @brief Reference to an MLIR Context
            mlir::MLIRContext &context;

            /// @brief Module to return on lifting process
            mlir::ModuleOp Module;

            /// @brief The builder is a helper class to create IR inside a function. The
            /// builder is stateful, in particular it keeps an "insertion point": this is
            /// where the next operations will be introduced.
            mlir::OpBuilder builder;

            /// @brief In case an instruction has some error while lifting it
            /// generate an exception or generate a NOP instruction
            bool gen_exception;

            /// @brief Method currently analyzed, must be updated for each analyzed method
            shuriken::analysis::dex::MethodAnalysis *current_method;
            /// @brief Basic block currently analyzed, must be updated for each basic
            /// block analyzed
            shuriken::analysis::dex::DVMBasicBlock *current_basic_block;
            /// @brief name of the module where we will write all the methods
            std::string module_name;

            // types from DVM for not generating it many times
            ::mlir::shuriken::MjolnIR::DVMVoidType voidType;
            ::mlir::shuriken::MjolnIR::DVMByteType byteType;
            ::mlir::shuriken::MjolnIR::DVMBoolType boolType;
            ::mlir::shuriken::MjolnIR::DVMCharType charType;
            ::mlir::shuriken::MjolnIR::DVMShortType shortType;
            ::mlir::shuriken::MjolnIR::DVMIntType intType;
            ::mlir::shuriken::MjolnIR::DVMLongType longType;
            ::mlir::shuriken::MjolnIR::DVMFloatType floatType;
            ::mlir::shuriken::MjolnIR::DVMDoubleType doubleType;
            ::mlir::shuriken::MjolnIR::DVMObjectType strObjectType;

            //===----------------------------------------------------------------------===//
            // Some generators methods
            //===----------------------------------------------------------------------===//
            /// @brief Return an mlir::Type from a Fundamental type of Dalvik
            /// @param fundamental fundamental type of Dalvik
            /// @return different type depending on input
            mlir::Type get_type(parser::dex::DVMFundamental *fundamental);

            /// @brief Return an mlir::Type from a class type of Dalvik
            /// @param cls class type of Dalvik
            /// @return a mlir::Type which contains as attribute the name of the class
            mlir::Type get_type(parser::dex::DVMClass *cls);

            /// @brief Generic generator method for all DVMType
            /// @param type type to obtain the mlir::Type
            /// @return an mlir::Type from the dalvik type
            mlir::Type get_type(parser::dex::DVMType *type);

            /// @brief Given a prototype generate the types in MLIR
            /// @param proto prototype of the method
            /// @return vector with the generated types from the parameters
            llvm::SmallVector<mlir::Type> gen_prototype(parser::dex::ProtoID *proto);

            /// @brief Generate a MethodOp from a EncodedMethod given
            /// @param method pointer to an encoded method to generate a MethodOp
            /// @return method operation from Dalvik
            ::mlir::shuriken::MjolnIR::MethodOp get_method(shuriken::analysis::dex::MethodAnalysis *M);

            //===----------------------------------------------------------------------===//
            // Lifting instructions, these class functions will be specialized for the
            // different function types.
            //===----------------------------------------------------------------------===//
            void gen_instruction(shuriken::analysis::dex::Instruction31c *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction31i *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction32x *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction22x *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction21c *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction35c *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction51l *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction21h *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction21s *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction11n *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction10x *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction10t *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction20t *instr);

            void gen_instruction(shuriken::analysis::dex::Instruction30t *instr);
            /// @brief Lift an instruction of the type Instruction23x
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction23x *instr);

            /// @brief Lift an instruction of the type Instruction12x
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction12x *instr);

            /// @brief Lift an instruction of type Instruction22s
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction22s *instr);

            /// @brief Lift an instruction of type Instruction22b
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction22b *instr);

            /// @brief Lift an instruction of type Instruction22t
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction22t *instr);

            /// @brief Lift an instruction of type Instruction21t
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction21t *instr);

            /// @brief Lift an instruction of type Instruction11x
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction11x *instr);

            /// @brief Lift an instruction of type Instruction22
            /// @param instr instruction to lift
            void gen_instruction(shuriken::analysis::dex::Instruction22c *instr);

            /// @brief Generate the IR from an instruction
            /// @param instr instruction from Dalvik to generate the IR
            void gen_instruction(shuriken::analysis::dex::Instruction *instr);

            /// @brief Generate a block into an mlir::Block*, we will lift each
            /// instruction.
            /// @param bb DVMBasicBlock to lift
            /// @param method method where the basic block is
            void gen_block(shuriken::analysis::dex::DVMBasicBlock *bb);

            void gen_terminators(shuriken::analysis::dex::DVMBasicBlock *bb);

            /// @brief Generate a MethodOp from a MethodAnalysis
            /// @param method MethodAnalysis object to lift
            void gen_method(shuriken::analysis::dex::MethodAnalysis *method);

            /// @brief Initialize possible used types and other necessary stuff
            void init();

        public:
            /// @brief Constructor of Lifter
            /// @param context context from MjolnIR
            /// @param gen_exception generate a exception or nop instruction
            Lifter(mlir::MLIRContext &context, bool gen_exception)
                : context(context), builder(&context), gen_exception(gen_exception) {
                init();
            }

            /// @brief Generate a ModuleOp with the lifted instructions from a
            /// MethodAnalysis
            /// @param methodAnalysis method analysis to lift to MjolnIR
            /// @return reference to ModuleOp with the lifted instructions
            mlir::OwningOpRef<mlir::ModuleOp>
            mlirGen(shuriken::analysis::dex::MethodAnalysis *methodAnalysis);
        };
    }// namespace MjolnIR
}// namespace shuriken

#endif// LIFTER_MJOLNIR_LIFTER_HPP
