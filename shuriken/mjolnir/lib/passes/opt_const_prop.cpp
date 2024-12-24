#include "passes/opt.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
namespace {

    namespace shuriken {
        namespace MjolnIR {
            namespace Opt {
                using namespace mlir;
                using namespace mlir::dataflow;

                /// INFO: A lattice value class to be wrapped around an Analysis State class for dataflow analysis
                class ConstantValue {
                public:
                    /// Construct an uninitialized const value.
                    explicit ConstantValue() = default;

                    /// Construct a known-constant const value.
                    explicit ConstantValue(Attribute constant, Dialect *dialect);

                    Attribute getConstantValue() const;

                    Dialect *getConstantDialect() const;

                    bool operator==(const ConstantValue &rhs) const;

                    void print(raw_ostream &os) const;

                    static ConstantValue getUninitialized();

                    bool isUninitialized() const;


                    static ConstantValue getUnknownConstant();

                    static ConstantValue join(const ConstantValue &lhs, const ConstantValue &rhs);

                private:
                    std::optional<Attribute> constant;
                    Dialect *dialect = nullptr;
                };
                class ConstProp : public SparseForwardDataFlowAnalysis<Lattice<ConstantValue>> {
                public:
                    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

                    void visitOperation(Operation *op,
                                        ArrayRef<const Lattice<ConstantValue> *> operands,
                                        ArrayRef<Lattice<ConstantValue> *>
                                                results) override;

                    void setToEntryState(Lattice<ConstantValue> *lattice) override;
                };


                [[maybe_unused]]
                mlir::LogicalResult const_prop(mlir::ModuleOp &module) {
                    auto solver = DataFlowSolver();
                    /*solver.load<ConstProp>();*/
                    /*[[maybe_unused]]*/
                    /*auto dataflow_result = solver.initializeAndRun(module);*/

                    return failure();
                }
            }// namespace Opt
        }// namespace MjolnIR
    }// namespace shuriken
}// namespace
