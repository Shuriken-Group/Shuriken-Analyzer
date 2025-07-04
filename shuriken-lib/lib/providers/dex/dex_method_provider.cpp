//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/providers/dex/dex_method_provider.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/class.hpp"


using namespace shuriken::dex;

namespace {
    std::string access_flags_to_string(uint32_t flags) {
        if (flags == types::access_flags::NONE) {
            return "NONE";
        }

        std::vector<std::string> flag_strings;

        // Check each flag in order of value
        if (flags & types::access_flags::ACC_PUBLIC) flag_strings.push_back("ACC_PUBLIC");
        if (flags & types::access_flags::ACC_PRIVATE) flag_strings.push_back("ACC_PRIVATE");
        if (flags & types::access_flags::ACC_PROTECTED) flag_strings.push_back("ACC_PROTECTED");
        if (flags & types::access_flags::ACC_STATIC) flag_strings.push_back("ACC_STATIC");
        if (flags & types::access_flags::ACC_FINAL) flag_strings.push_back("ACC_FINAL");
        if (flags & types::access_flags::ACC_SYNCHRONIZED) flag_strings.push_back("ACC_SYNCHRONIZED");

        // Handle overlapping values - check context or prioritize
        if (flags & 0x40) {
            // Both ACC_VOLATILE and ACC_BRIDGE have the same value
            // You might want to add logic to distinguish based on context
            flag_strings.push_back("ACC_VOLATILE/ACC_BRIDGE");
        }

        if (flags & 0x80) {
            // Both ACC_TRANSIENT and ACC_VARARGS have the same value
            flag_strings.push_back("ACC_TRANSIENT/ACC_VARARGS");
        }

        if (flags & types::access_flags::ACC_NATIVE) flag_strings.push_back("ACC_NATIVE");
        if (flags & types::access_flags::ACC_INTERFACE) flag_strings.push_back("ACC_INTERFACE");
        if (flags & types::access_flags::ACC_ABSTRACT) flag_strings.push_back("ACC_ABSTRACT");
        if (flags & types::access_flags::ACC_STRICT) flag_strings.push_back("ACC_STRICT");
        if (flags & types::access_flags::ACC_SYNTHETIC) flag_strings.push_back("ACC_SYNTHETIC");
        if (flags & types::access_flags::ACC_ANNOTATION) flag_strings.push_back("ACC_ANNOTATION");
        if (flags & types::access_flags::ACC_ENUM) flag_strings.push_back("ACC_ENUM");
        if (flags & types::access_flags::UNUSED) flag_strings.push_back("UNUSED");
        if (flags & types::access_flags::ACC_CONSTRUCTOR) flag_strings.push_back("ACC_CONSTRUCTOR");
        if (flags & types::access_flags::ACC_DECLARED_SYNCHRONIZED) flag_strings.push_back("ACC_DECLARED_SYNCHRONIZED");

        // Join with pipes
        std::string result;
        for (size_t i = 0; i < flag_strings.size(); ++i) {
            if (i > 0) result += "|";
            result += flag_strings[i];
        }

        return result.empty() ? "NONE" : result;
    }

// Template factory function for creating instruction wrappers
    template<typename WrapperType, typename ProviderType>
    std::unique_ptr<Instruction> get_instruction(ProviderType &provider) {
        return std::make_unique<WrapperType>(provider);
    }

// Type alias for the factory function
    using instruction_factory_func = std::function<std::unique_ptr<Instruction>(void *)>;

// Wrapper function to handle the void* casting
    template<typename WrapperType, typename ProviderType>
    std::unique_ptr<Instruction> create_wrapper(void *provider_ptr) {
        auto *typed_provider = static_cast<ProviderType *>(provider_ptr);
        return std::make_unique<WrapperType>(*typed_provider);
    }

// Create the instruction wrapper mapping using the .def file
    inline std::unordered_map<disassembler::opcodes, instruction_factory_func> create_instruction_wrapper_mappings() {
        return {
                // Instruction00x mappings
                {disassembler::opcodes::OP_IGET_VOLATILE,              &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IPUT_VOLATILE,              &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_SGET_VOLATILE,              &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_SPUT_VOLATILE,              &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IGET_OBJECT_VOLATILE,       &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IGET_WIDE_VOLATILE,         &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IPUT_WIDE_VOLATILE,         &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_SGET_WIDE_VOLATILE,         &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_SPUT_WIDE_VOLATILE,         &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_BREAKPOINT,                 &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_THROW_VERIFICATION_ERROR,   &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_EXECUTE_INLINE,             &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_EXECUTE_INLINE_RANGE,       &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_INVOKE_OBJECT_INIT_RANGE,   &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_RETURN_VOID_BARRIER,        &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IGET_QUICK,                 &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IGET_WIDE_QUICK,            &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IGET_OBJECT_QUICK,          &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IPUT_QUICK,                 &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IPUT_WIDE_QUICK,            &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_IPUT_OBJECT_QUICK,          &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_INVOKE_VIRTUAL_QUICK,       &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_INVOKE_VIRTUAL_QUICK_RANGE, &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_3E,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_3F,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_40,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_41,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_42,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_43,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_73,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_79,                  &create_wrapper<Instruction00x, Instruction00xProvider>},
                {disassembler::opcodes::OP_UNUSED_7A,                  &create_wrapper<Instruction00x, Instruction00xProvider>},

                // Instruction10x mappings
                {disassembler::opcodes::OP_RETURN_VOID,                &create_wrapper<Instruction10x, Instruction10xProvider>},
                {disassembler::opcodes::OP_NOP,                        &create_wrapper<Instruction10x, Instruction10xProvider>},

                // Instruction11n mappings
                {disassembler::opcodes::OP_CONST_4,                    &create_wrapper<Instruction11n, Instruction11nProvider>},

                // Instruction11x mappings
                {disassembler::opcodes::OP_MOVE_RESULT,                &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_MOVE_RESULT_WIDE,           &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_MOVE_RESULT_OBJECT,         &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_MOVE_EXCEPTION,             &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_RETURN,                     &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_RETURN_WIDE,                &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_RETURN_OBJECT,              &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_MONITOR_ENTER,              &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_MONITOR_EXIT,               &create_wrapper<Instruction11x, Instruction11xProvider>},
                {disassembler::opcodes::OP_THROW,                      &create_wrapper<Instruction11x, Instruction11xProvider>},

                // Instruction12x mappings
                {disassembler::opcodes::OP_MOVE,                       &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_MOVE_WIDE,                  &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_MOVE_OBJECT,                &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_ARRAY_LENGTH,               &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_NEG_INT,                    &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_NOT_INT,                    &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_NEG_LONG,                   &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_NOT_LONG,                   &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_NEG_FLOAT,                  &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_NEG_DOUBLE,                 &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_INT_TO_LONG,                &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_INT_TO_FLOAT,               &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_INT_TO_DOUBLE,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_LONG_TO_INT,                &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_LONG_TO_FLOAT,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_LONG_TO_DOUBLE,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_FLOAT_TO_INT,               &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_FLOAT_TO_LONG,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_FLOAT_TO_DOUBLE,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DOUBLE_TO_INT,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DOUBLE_TO_LONG,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DOUBLE_TO_FLOAT,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_INT_TO_BYTE,                &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_INT_TO_CHAR,                &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_INT_TO_SHORT,               &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_ADD_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SUB_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_MUL_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DIV_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_REM_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_AND_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_OR_INT_2ADDR,               &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_XOR_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SHL_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SHR_INT_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_USHR_INT_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_ADD_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SUB_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_MUL_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DIV_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_REM_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_AND_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_OR_LONG_2ADDR,              &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_XOR_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SHL_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SHR_LONG_2ADDR,             &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_USHR_LONG_2ADDR,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_ADD_FLOAT_2ADDR,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SUB_FLOAT_2ADDR,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_MUL_FLOAT_2ADDR,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DIV_FLOAT_2ADDR,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_REM_FLOAT_2ADDR,            &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_ADD_DOUBLE_2ADDR,           &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_SUB_DOUBLE_2ADDR,           &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_MUL_DOUBLE_2ADDR,           &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_DIV_DOUBLE_2ADDR,           &create_wrapper<Instruction12x, Instruction12xProvider>},
                {disassembler::opcodes::OP_REM_DOUBLE_2ADDR,           &create_wrapper<Instruction12x, Instruction12xProvider>},

                // Instruction10t mappings
                {disassembler::opcodes::OP_GOTO,                       &create_wrapper<Instruction10t, Instruction10tProvider>},

                // Instruction20t mappings
                {disassembler::opcodes::OP_GOTO_16,                    &create_wrapper<Instruction20t, Instruction20tProvider>},

                // Instruction21h mappings
                {disassembler::opcodes::OP_CONST_HIGH16,               &create_wrapper<Instruction21h, Instruction21hProvider>},
                {disassembler::opcodes::OP_CONST_WIDE_HIGH16,          &create_wrapper<Instruction21h, Instruction21hProvider>},

                // Instruction21s mappings
                {disassembler::opcodes::OP_CONST_16,                   &create_wrapper<Instruction21s, Instruction21sProvider>},
                {disassembler::opcodes::OP_CONST_WIDE_16,              &create_wrapper<Instruction21s, Instruction21sProvider>},

                // Instruction21t mappings
                {disassembler::opcodes::OP_IF_EQZ,                     &create_wrapper<Instruction21t, Instruction21tProvider>},
                {disassembler::opcodes::OP_IF_NEZ,                     &create_wrapper<Instruction21t, Instruction21tProvider>},
                {disassembler::opcodes::OP_IF_LTZ,                     &create_wrapper<Instruction21t, Instruction21tProvider>},
                {disassembler::opcodes::OP_IF_GEZ,                     &create_wrapper<Instruction21t, Instruction21tProvider>},
                {disassembler::opcodes::OP_IF_GTZ,                     &create_wrapper<Instruction21t, Instruction21tProvider>},
                {disassembler::opcodes::OP_IF_LEZ,                     &create_wrapper<Instruction21t, Instruction21tProvider>},

                // Instruction22x mappings
                {disassembler::opcodes::OP_MOVE_FROM16,                &create_wrapper<Instruction22x, Instruction22xProvider>},
                {disassembler::opcodes::OP_MOVE_WIDE_FROM16,           &create_wrapper<Instruction22x, Instruction22xProvider>},
                {disassembler::opcodes::OP_MOVE_OBJECT_FROM16,         &create_wrapper<Instruction22x, Instruction22xProvider>},

                // Instruction22b mappings
                {disassembler::opcodes::OP_ADD_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_SUB_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_MUL_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_DIV_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_REM_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_AND_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_OR_INT_LIT8,                &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_XOR_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_SHL_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_SHR_INT_LIT8,               &create_wrapper<Instruction22b, Instruction22bProvider>},
                {disassembler::opcodes::OP_USHR_INT_LIT8,              &create_wrapper<Instruction22b, Instruction22bProvider>},

                // Instruction22c mappings
                {disassembler::opcodes::OP_INSTANCE_OF,                &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_NEW_ARRAY,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET,                       &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET_WIDE,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET_OBJECT,                &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET_BOOLEAN,               &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET_BYTE,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET_CHAR,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IGET_SHORT,                 &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT,                       &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT_WIDE,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT_OBJECT,                &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT_BOOLEAN,               &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT_BYTE,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT_CHAR,                  &create_wrapper<Instruction22c, Instruction22cProvider>},
                {disassembler::opcodes::OP_IPUT_SHORT,                 &create_wrapper<Instruction22c, Instruction22cProvider>},

                // Instruction22s mappings
                {disassembler::opcodes::OP_ADD_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_SUB_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_MUL_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_DIV_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_REM_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_AND_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_OR_INT_LIT16,               &create_wrapper<Instruction22s, Instruction22sProvider>},
                {disassembler::opcodes::OP_XOR_INT_LIT16,              &create_wrapper<Instruction22s, Instruction22sProvider>},

                // Instruction22t mappings
                {disassembler::opcodes::OP_IF_EQ,                      &create_wrapper<Instruction22t, Instruction22tProvider>},
                {disassembler::opcodes::OP_IF_NE,                      &create_wrapper<Instruction22t, Instruction22tProvider>},
                {disassembler::opcodes::OP_IF_LT,                      &create_wrapper<Instruction22t, Instruction22tProvider>},
                {disassembler::opcodes::OP_IF_GE,                      &create_wrapper<Instruction22t, Instruction22tProvider>},
                {disassembler::opcodes::OP_IF_GT,                      &create_wrapper<Instruction22t, Instruction22tProvider>},
                {disassembler::opcodes::OP_IF_LE,                      &create_wrapper<Instruction22t, Instruction22tProvider>},

                // Instruction23x mappings
                {disassembler::opcodes::OP_CMPL_FLOAT,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_CMPG_FLOAT,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_CMPL_DOUBLE,                &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_CMPG_DOUBLE,                &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_CMP_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_ADD_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SUB_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_MUL_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_DIV_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_REM_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AND_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_OR_INT,                     &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_XOR_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SHL_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SHR_INT,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_USHR_INT,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_ADD_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SUB_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_MUL_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_DIV_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_REM_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AND_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_OR_LONG,                    &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_XOR_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SHL_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SHR_LONG,                   &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_USHR_LONG,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_ADD_FLOAT,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SUB_FLOAT,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_MUL_FLOAT,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_DIV_FLOAT,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_REM_FLOAT,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_ADD_DOUBLE,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_SUB_DOUBLE,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_MUL_DOUBLE,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_DIV_DOUBLE,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_REM_DOUBLE,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET,                       &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET_WIDE,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET_OBJECT,                &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET_BOOLEAN,               &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET_BYTE,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET_CHAR,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_AGET_SHORT,                 &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT,                       &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT_WIDE,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT_OBJECT,                &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT_BOOLEAN,               &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT_BYTE,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT_CHAR,                  &create_wrapper<Instruction23x, Instruction23xProvider>},
                {disassembler::opcodes::OP_APUT_SHORT,                 &create_wrapper<Instruction23x, Instruction23xProvider>},

                // Instruction30t mappings
                {disassembler::opcodes::OP_GOTO_32,                    &create_wrapper<Instruction30t, Instruction30tProvider>},

                // Instruction31i mappings
                {disassembler::opcodes::OP_CONST,                      &create_wrapper<Instruction31i, Instruction31iProvider>},
                {disassembler::opcodes::OP_CONST_WIDE_32,              &create_wrapper<Instruction31i, Instruction31iProvider>},

                // Instruction31c mappings
                {disassembler::opcodes::OP_CONST_STRING_JUMBO,         &create_wrapper<Instruction31c, Instruction31cProvider>},

                // Instruction31t mappings
                {disassembler::opcodes::OP_FILL_ARRAY_DATA,            &create_wrapper<Instruction31t, Instruction31tProvider>},
                {disassembler::opcodes::OP_PACKED_SWITCH,              &create_wrapper<Instruction31t, Instruction31tProvider>},
                {disassembler::opcodes::OP_SPARSE_SWITCH,              &create_wrapper<Instruction31t, Instruction31tProvider>},

                // Instruction32x mappings
                {disassembler::opcodes::OP_MOVE_16,                    &create_wrapper<Instruction32x, Instruction32xProvider>},
                {disassembler::opcodes::OP_MOVE_WIDE_16,               &create_wrapper<Instruction32x, Instruction32xProvider>},
                {disassembler::opcodes::OP_MOVE_OBJECT_16,             &create_wrapper<Instruction32x, Instruction32xProvider>},

                // Instruction21c mappings
                {disassembler::opcodes::OP_CONST_STRING,               &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_CONST_CLASS,                &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_CHECK_CAST,                 &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_NEW_INSTANCE,               &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET,                       &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET_WIDE,                  &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET_OBJECT,                &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET_BOOLEAN,               &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET_BYTE,                  &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET_CHAR,                  &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SGET_SHORT,                 &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT,                       &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_WIDE,                  &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_OBJECT,                &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_BOOLEAN,               &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_BYTE,                  &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_CHAR,                  &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_SHORT,                 &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_SPUT_OBJECT_VOLATILE,       &create_wrapper<Instruction21c, Instruction21cProvider>},
                {disassembler::opcodes::OP_CONST_METHOD_TYPE,          &create_wrapper<Instruction21c, Instruction21cProvider>},

                // Instruction35c mappings
                {disassembler::opcodes::OP_FILLED_NEW_ARRAY,           &create_wrapper<Instruction35c, Instruction35cProvider>},
                {disassembler::opcodes::OP_INVOKE_VIRTUAL,             &create_wrapper<Instruction35c, Instruction35cProvider>},
                {disassembler::opcodes::OP_INVOKE_SUPER,               &create_wrapper<Instruction35c, Instruction35cProvider>},
                {disassembler::opcodes::OP_INVOKE_DIRECT,              &create_wrapper<Instruction35c, Instruction35cProvider>},
                {disassembler::opcodes::OP_INVOKE_STATIC,              &create_wrapper<Instruction35c, Instruction35cProvider>},
                {disassembler::opcodes::OP_INVOKE_INTERFACE,           &create_wrapper<Instruction35c, Instruction35cProvider>},
                {disassembler::opcodes::OP_IPUT_OBJECT_VOLATILE,       &create_wrapper<Instruction35c, Instruction35cProvider>},

                // Instruction3rc mappings
                {disassembler::opcodes::OP_FILLED_NEW_ARRAY_RANGE,     &create_wrapper<Instruction3rc, Instruction3rcProvider>},
                {disassembler::opcodes::OP_INVOKE_VIRTUAL_RANGE,       &create_wrapper<Instruction3rc, Instruction3rcProvider>},
                {disassembler::opcodes::OP_INVOKE_SUPER_RANGE,         &create_wrapper<Instruction3rc, Instruction3rcProvider>},
                {disassembler::opcodes::OP_INVOKE_DIRECT_RANGE,        &create_wrapper<Instruction3rc, Instruction3rcProvider>},
                {disassembler::opcodes::OP_INVOKE_STATIC_RANGE,        &create_wrapper<Instruction3rc, Instruction3rcProvider>},
                {disassembler::opcodes::OP_INVOKE_INTERFACE_RANGE,     &create_wrapper<Instruction3rc, Instruction3rcProvider>},
                {disassembler::opcodes::OP_SGET_OBJECT_VOLATILE,       &create_wrapper<Instruction3rc, Instruction3rcProvider>},

                // Instruction45cc mappings
                {disassembler::opcodes::OP_INVOKE_SUPER_QUICK,         &create_wrapper<Instruction45cc, Instruction45ccProvider>},

                // Instruction4rcc mappings
                {disassembler::opcodes::OP_INVOKE_SUPER_QUICK_RANGE,   &create_wrapper<Instruction4rcc, Instruction4rccProvider>},

                // Instruction51l mappings
                {disassembler::opcodes::OP_CONST_WIDE,                 &create_wrapper<Instruction51l, Instruction51lProvider>},
        };
    }

// Global accessor function (thread-safe singleton pattern)
    inline const auto &get_instruction_wrapper_mappings() {
        static const auto mappings = create_instruction_wrapper_mappings();
        return mappings;
    }

// Convenience function to create instruction wrapper from opcode and provider
    inline std::unique_ptr<Instruction> create_instruction_wrapper(disassembler::opcodes opcode, void *provider_ptr) {
        const auto &mappings = get_instruction_wrapper_mappings();
        auto it = mappings.find(opcode);
        if (it != mappings.end()) {
            return it->second(provider_ptr);
        }
        return nullptr; // Unknown opcode
    }

// Type-safe template version
    template<typename ProviderType>
    std::unique_ptr<Instruction>
    create_instruction_wrapper_typed(disassembler::opcodes opcode, ProviderType &provider) {
        return create_instruction_wrapper(opcode, &provider);
    }
}

shuriken::dex::DexMethodProvider::DexMethodProvider(std::string_view name,
                                                    types::access_flags access_flags,
                                                    DVMPrototype &method_prototype,
                                                    types::method_type_e method_type,
                                                    Class &owner_class,
                                                    Dex &owner_dex,
                                                    DexEngine &dex_engine,
                                                    std::uint16_t number_of_registers,
                                                    std::vector<std::uint8_t> &bytecode,
                                                    EncodedMethod *method)
        : method_name(name),
          method_access_flags(access_flags), method_prototype(method_prototype),
          method_type(method_type),
          owner_class(owner_class), owner_dex(owner_dex),
          dex_engine(dex_engine),
          number_of_registers(number_of_registers),
          bytecode(std::move(bytecode)),
          method(method) {
    method_descriptor = owner_class.get_name_string() + "->"
                        + std::string(name) + method_prototype.get_descriptor_string();
}


std::string_view shuriken::dex::DexMethodProvider::get_name() const {
    return method_name;
}


std::string shuriken::dex::DexMethodProvider::get_name_string() const {
    return method_name;
}


shuriken::dex::types::access_flags shuriken::dex::DexMethodProvider::get_method_access_flags() const {
    return method_access_flags;
}

std::string_view shuriken::dex::DexMethodProvider::get_method_access_flags_str() {
    if (access_flags_str.empty()) {
        access_flags_str = ::access_flags_to_string(method_access_flags);
    }
    return access_flags_str;
}

const shuriken::dex::DVMPrototype &shuriken::dex::DexMethodProvider::get_method_prototype() const {
    return method_prototype;
}


shuriken::dex::DVMPrototype &shuriken::dex::DexMethodProvider::get_method_prototype() {
    return method_prototype;
}

types::method_type_e shuriken::dex::DexMethodProvider::get_method_type() const {
    return method_type;
}


const shuriken::dex::Class &shuriken::dex::DexMethodProvider::get_owner_class() const {
    return owner_class;
}


shuriken::dex::Class &shuriken::dex::DexMethodProvider::get_owner_class() {
    return owner_class;
}

const shuriken::dex::Dex &shuriken::dex::DexMethodProvider::get_owner_dex() const {
    return owner_dex;
}


shuriken::dex::Dex &shuriken::dex::DexMethodProvider::get_owner_dex() {
    return owner_dex;
}


std::string_view shuriken::dex::DexMethodProvider::get_descriptor() const {
    return method_descriptor;
}


std::string shuriken::dex::DexMethodProvider::get_descriptor_string() const {
    return method_descriptor;
}


std::uint16_t shuriken::dex::DexMethodProvider::registers_size() const {
    return number_of_registers;
}


std::span<const std::uint8_t> shuriken::dex::DexMethodProvider::get_bytecode() const {
    static std::span<const std::uint8_t> data{bytecode};
    return data;
}

std::vector<std::uint8_t> &shuriken::dex::DexMethodProvider::get_bytecode_vector() {
    return bytecode;
}

EncodedMethod *shuriken::dex::DexMethodProvider::get_encoded_method() const {
    return method;
}

void DexMethodProvider::set_method_instructions(std::list<std::unique_ptr<InstructionProvider>> &instructions) {
    this->instructions = std::move(instructions);

    std::unordered_map<InstructionProvider *, Instruction *> mapping_switch;
    for (const auto &instr: this->instructions) {
        instructions_usr.push_back(::create_instruction_wrapper(instr->get_instruction_opcode(), instr.get()));
        instructions_usr_r.push_back(*instructions_usr.back());
        if (instr->get_instruction_opcode() == disassembler::opcodes::OP_PACKED_SWITCH_TABLE ||
            instr->get_instruction_opcode() == disassembler::opcodes::OP_SPARSE_SWITCH_TABLE) {
            mapping_switch[instr.get()] = instructions_usr.back().get();
        }
    }

    if (!mapping_switch.empty()) {
        for (auto &instr: this->instructions) {
            if (instr->get_instruction_opcode() == disassembler::opcodes::OP_PACKED_SWITCH) {
                auto *switch_instr = reinterpret_cast<Instruction31tProvider *>(instr.get());
                InstructionProvider *i = std::get<PackedSwitchProvider *>(switch_instr->get_switch());
                switch_instr->set_packed_switch_usr(reinterpret_cast<PackedSwitch *>(mapping_switch[i]));
            } else if (instr->get_instruction_opcode() == disassembler::opcodes::OP_SPARSE_SWITCH) {
                auto *switch_instr = reinterpret_cast<Instruction31tProvider *>(instr.get());
                InstructionProvider *i = std::get<SparseSwitchProvider *>(switch_instr->get_switch());
                switch_instr->set_sparse_switch_usr(reinterpret_cast<SparseSwitch *>(mapping_switch[i]));
            }
        }
    }
}

void DexMethodProvider::set_exceptions(disassembler::exceptions_data_t &exceptions) {
    this->exceptions = std::move(exceptions);
}

std::list<std::reference_wrapper<Instruction>> &DexMethodProvider::get_method_instructions() {
    if (!disassembled) {
        this->dex_engine.get().disassemble_method(*this);
        disassembled = true;
    }
    return instructions_usr_r;
}

disassembler::exceptions_data_t &DexMethodProvider::get_exceptions() {
    if (!disassembled) {
        this->dex_engine.get().disassemble_method(*this);
        disassembled = true;
    }
    return exceptions;
}




