#include "transform/mjolnir_to_smali.h"
#include <cstdlib>
#include <iostream>
#include <regex>

namespace {
    std::string get_suffix_field_instr(mlir::Value reg) {
        mlir::Type type = reg.getType();

        // Check for IntegerType and get its width
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
            unsigned width = intType.getWidth();
            auto signedness = intType.getSignedness();
            if (width == 1) {
                return "-boolean";
            } else if (width == 8 && signedness == mlir::IntegerType::Signed) {
                return "-byte";
            } else if (width == 8 && signedness == mlir::IntegerType::Unsigned) {
                return "-char";
            } else if (width == 16) {
                return "-short";
            } else if (width == 32) {
                return "";
            } else if (width == 64) {
                return "-wide";
            }
        }

        // Check for Float32
        if (mlir::isa<mlir::Float32Type>(type)) {
            return "";
        }

        // Check for Float64
        if (mlir::isa<mlir::Float64Type>(type)) {
            return "-wide";
        }

        // Check for your custom DVMObjectType
        if (mlir::isa<::mlir::shuriken::MjolnIR::DVMObjectType>(type)) {
            return "-object";
        }

        return "";
    }

    std::string get_type_for_field(mlir::Value reg) {
        mlir::Type type = reg.getType();

        // Check for IntegerType and get its width
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
            unsigned width = intType.getWidth();
            auto signedness = intType.getSignedness();
            if (width == 1) {
                return "Z";  // boolean
            } else if (width == 8 && signedness == mlir::IntegerType::Signed) {
                return "B";  // byte
            } else if (width == 8 && signedness == mlir::IntegerType::Unsigned) {
                return "C";  // char
            } else if (width == 16) {
                return "S";  // short
            } else if (width == 32) {
                return "I";  // int
            } else if (width == 64) {
                return "J";  // long
            }
        }

        // Check for Float32
        if (mlir::isa<mlir::Float32Type>(type)) {
            return "F";  // float
        }

        // Check for Float64
        if (mlir::isa<mlir::Float64Type>(type)) {
            return "D";  // double
        }

        // Check for your custom DVMObjectType
        if (mlir::isa<::mlir::shuriken::MjolnIR::DVMObjectType>(type)) {
            return "L";  // object
        }

        return "";  // default case
    }

    std::string get_dalviktype_from_mlir_type(mlir::Type type) {
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
            unsigned width = intType.getWidth();
            auto signedness = intType.getSignedness();
            if (width == 1) {
                return "Z";  // boolean
            } else if (width == 8 && signedness == mlir::IntegerType::Signed) {
                return "B";  // byte
            } else if (width == 8 && signedness == mlir::IntegerType::Unsigned) {
                return "C";  // char
            } else if (width == 16) {
                return "S";  // short
            } else if (width == 32) {
                return "I";  // int
            } else if (width == 64) {
                return "J";  // long
            }
        }

        // Check for Float32
        if (mlir::isa<mlir::Float32Type>(type)) {
            return "F";  // float
        }

        // Check for Float64
        if (mlir::isa<mlir::Float64Type>(type)) {
            return "D";  // double
        }

        if (mlir::isa<::mlir::shuriken::MjolnIR::DVMVoidType>(type)) {
            return "V";
        }

        if (auto objectType = mlir::dyn_cast<::mlir::shuriken::MjolnIR::DVMObjectType>(type)) {
            auto cls_name = objectType.getValue();
            if (cls_name.starts_with("L") && cls_name.ends_with(";"))
                return cls_name.str();
            return fmt::format("L{};", cls_name.data());
        }

        if (auto arrayType = mlir::dyn_cast<::mlir::shuriken::MjolnIR::DVMArrayType>(type)) {
            auto type_name = arrayType.getArrayType();
            return type_name.str();
        }

        return "ERROR";
    }
}

namespace shuriken::MjolnIR {
    /// INFO: MJOLNIR
    std::tuple<SmaliLines, SmaliLines> MjolnIRToSmali::from_mjolnir_method_op(MethodOp op) {
        // for every method, reset the counter
        vrc.clean_counter();

        // extract the whole method information
        auto method_name = op.getName().data();
        auto parameter_types = op.getArgumentTypes();
        auto ret_type = op.getResultTypes();


        std::stringstream p;
        bool first = true;

        // Access modifiers first
        if (op.isPublic()) {
            p << "public";
            first = false;
        }
        if (op.isPrivate()) {
            if (!first) p << " ";
            p << "private";
            first = false;
        }
        if (op.isProtected()) {
            if (!first) p << " ";
            p << "protected";
            first = false;
        }

        // Then static
        if (op.isStatic()) {
            if (!first) p << " ";
            p << "static";
            first = false;
        }

        // Then final
        if (op.isFinal()) {
            if (!first) p << " ";
            p << "final";
            first = false;
        }

        // Then synchronized
        if (op.isSynchronized()) {
            if (!first) p << " ";
            p << "synchronized";
            first = false;
        }

        std::stringstream parameters;
        parameters << "(";
        for (auto parameter_type : parameter_types)
            parameters << ::get_dalviktype_from_mlir_type(parameter_type);
        parameters << ")";


        SmaliLine prologue_line = fmt::format(".method {} {}{}{}", 
            p.str(),
            method_name, 
            parameters.str(), 
            ::get_dalviktype_from_mlir_type(ret_type[0]));


        SmaliLines prologue = {prologue_line};
        SmaliLines epilogue = {".end method"};
        return {prologue, epilogue};
    }
    SmaliLine MjolnIRToSmali::from_mjolnir_return_op(ReturnOp op) {
        auto operands = op.getOperands();
        if (operands.size() == 0) {
            return "return";
        } else if (operands.size() == 1) {
            return fmt::format("return {}", get_smali_value(operands[0]));
        }
        std::cerr << "Returning more than 1 operand, which is an impossible variant\n";
        std::abort();
    }

    SmaliLine MjolnIRToSmali::from_mjolnir_fallthrough(FallthroughOp) { return "nop"; }

    SmaliLine MjolnIRToSmali::from_mjolnir_loadfield(LoadFieldOp lfop) {
        auto access_type = lfop.getAccessType();
        auto field_name = lfop.getFieldName().data();
        auto field_class = lfop.getFieldClass().data();
        auto return_value = lfop.getResult();

        if (access_type == FieldAccessType::STATIC) {
            return fmt::format("sget{} {}, {}->{}:{}", 
                ::get_suffix_field_instr(return_value), 
                get_smali_value(return_value),
                field_class,
                field_name,
                ::get_type_for_field(return_value));
        } else {
            auto instance_register = get_smali_value(lfop.getInstance());
            return fmt::format("iget{} {}, {}, {}->{}:{}",
                ::get_suffix_field_instr(return_value),
                get_smali_value(return_value),
                instance_register,
                field_class,
                field_name,
                ::get_type_for_field(return_value));
        }
    }

    SmaliLine MjolnIRToSmali::from_mjolnir_storefield(StoreFieldOp sfop) { 
        auto access_type = sfop.getAccessType();
        auto field_name = sfop.getFieldName().data();
        auto field_class = sfop.getFieldClass().data();
        auto stored_value = sfop.getValue();

        if (access_type == FieldAccessType::STATIC) {
            return fmt::format("sput{} {}, {}->{}:{}",
                ::get_suffix_field_instr(stored_value),
                get_smali_value(stored_value),
                field_class, field_name,
                ::get_type_for_field(stored_value));
        } else {
            auto instance_register = get_smali_value(sfop.getInstance());
            return fmt::format("iput{} {}, {}, {}->{}:{}",
                ::get_suffix_field_instr(stored_value),
                get_smali_value(stored_value), instance_register,
                field_class, field_name,
                ::get_type_for_field(stored_value));
        }
    }
    
    SmaliLine MjolnIRToSmali::from_mjolnir_loadvalue(LoadValue) { return ""; }


    SmaliLine MjolnIRToSmali::from_mjolnir_move(MoveOp op) {
        auto dest = op.getResult();
        auto operand = op.getOperand();


        return fmt::format("move {}, {}", get_smali_value(dest), get_smali_value(operand));
    }
    SmaliLine MjolnIRToSmali::from_mjolnir_invoke(InvokeOp op) {
        auto class_name = op.getClassOwner().data();
        auto callee = op.getCallee().data();
        auto attribute = op.getInvokeType();
        auto inputs = op.getInputs();
        auto result = op.getResults();
        bool skipFirstParameter = true;
        
        std::string invoke_type;

        switch (attribute)
        {
        case mlir::shuriken::MjolnIR::InvokeType::DIRECT:
            invoke_type = "-direct";
            break;
        case mlir::shuriken::MjolnIR::InvokeType::INTERFACE:
            invoke_type = "-interface";
            break;
        case mlir::shuriken::MjolnIR::InvokeType::STATIC:
            skipFirstParameter = false;
            invoke_type = "-static";
            break;
        case mlir::shuriken::MjolnIR::InvokeType::SUPER:
            invoke_type = "-super";
            break;
        case mlir::shuriken::MjolnIR::InvokeType::VIRTUAL:
            invoke_type = "-virtual";
            break;
        default:
            break;
        }



        std::string parameters;
        std::string parameterTypes;
        parameters = "{";
        parameterTypes = "(";        
        int i = 0;
        for (auto input : inputs) {
            parameters += get_smali_value(input);
            parameters +=  ",";
            if (i==0 && skipFirstParameter) {
                i++;
                continue;
            }
            parameterTypes += ::get_dalviktype_from_mlir_type(input.getType());
        }
        if (!inputs.empty())
            parameters.pop_back();
        parameters += "}";
        parameterTypes += ")";

        std::string ret_type = result.size() == 0 ? "V" : ::get_dalviktype_from_mlir_type(result[0].getType());
        
        std::string move_result;

        if (result.size() != 0)
            move_result = "\n    move-result " + get_smali_value(result[0]);


        return fmt::format("invoke{} {}, {}->{}{}{} {}", invoke_type, parameters, class_name, callee, parameterTypes, ret_type, move_result);
    }
    SmaliLine MjolnIRToSmali::from_mjolnir_new(NewOp no) {
        auto ret_value = no.getResult();
        auto ret_value_type = mlir::dyn_cast<::mlir::shuriken::MjolnIR::DVMObjectType>(ret_value.getType());
        auto cls_value = ret_value_type.getValue().data();
        
        return fmt::format("new-instance {}, L{};", get_smali_value(ret_value), cls_value); 
    }
    SmaliLine MjolnIRToSmali::from_mjolnir_getarray(GetArrayOp) { return ""; }

    SmaliLine MjolnIRToSmali::from_mjolnir_loadstring(LoadString ls) {
        auto value = ls.getString().str();
        auto result = get_smali_value(ls.getResult());

        // First escape backslashes in the string
        auto escaped = std::regex_replace(value, std::regex("\n"), "\\n");
        return fmt::format("const-string {}, \"{}\"", result, escaped);
    }

}// namespace shuriken::MjolnIR
