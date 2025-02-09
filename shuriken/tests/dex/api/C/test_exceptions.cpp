#include <cassert>
#include <cstring>
#include <iostream>
#include "dex-files-folder.inc"
#include "shuriken/api/C/shuriken_core.h"

const char *file = DEX_FILES_FOLDER
    "exception_test.dex";

int main() {
    hDexContext context = parse_dex(file);
    assert(context != nullptr && "Failed to parse DEX file");

    // Get the test class
    hdvmclass_t* test_class = get_class_by_name(context, "ExceptionTest");
    assert(test_class != nullptr && "Failed to get ExceptionTest class");

    // First disassemble the DEX
    disassemble_dex(context);

    // Find the method with try/catch blocks
    hdvmmethod_t* try_catch_method = get_method_by_name(context, "LExceptionTest;->methodWithTryCatch()V");
    assert(try_catch_method != nullptr && "Failed to get method with try/catch");

    // Get disassembled method to analyze exceptions
    dvmdisassembled_method_t* disassembled = get_disassembled_method(context, try_catch_method->dalvik_name);
    assert(disassembled != nullptr && "Failed to get disassembled method");
    assert(disassembled->n_of_exceptions > 0 && "No exception handlers found");

    // Check exception information
    for (size_t i = 0; i < disassembled->n_of_exceptions; i++) {
        dvmexceptions_data_t* exception = &disassembled->exception_information[i];
        assert(exception->try_value_start_addr < exception->try_value_end_addr && 
               "Invalid try block addresses");
        assert(exception->n_of_handlers > 0 && "No handlers in exception data");
        
        // Check handlers
        for (size_t j = 0; j < exception->n_of_handlers; j++) {
            dvmhandler_data_t* handler = &exception->handler[j];
            assert(handler->handler_type != nullptr && "Handler type is null");
            assert(handler->handler_start_addr > 0 && "Invalid handler start address");
        }
    }

    // Create analysis for basic blocks
    create_dex_analysis(context, true);
    analyze_classes(context);

    // Get method analysis to check basic blocks
    hdvmmethodanalysis_t* method_analysis = get_analyzed_method_by_hdvmmethod(context, try_catch_method);
    assert(method_analysis != nullptr && "Failed to get method analysis");
    assert(method_analysis->basic_blocks != nullptr && "No basic blocks in analysis");
    assert(method_analysis->basic_blocks->n_of_blocks > 0 && "Empty basic blocks");

    // Check basic blocks structure
    bool found_try_block = false;
    bool found_catch_block = false;

    for (size_t i = 0; i < method_analysis->basic_blocks->n_of_blocks; i++) {
        hdvmbasicblock_t* block = &method_analysis->basic_blocks->blocks[i];
        assert(block->name != nullptr && "Basic block name is null");
        assert(block->block_string != nullptr && "Basic block string is null");
        assert(block->n_of_instructions > 0 && "Empty basic block");
        assert(block->instructions != nullptr && "No instructions in block");

        if (block->try_block) {
            found_try_block = true;
        }
        if (block->catch_block) {
            found_catch_block = true;
            assert(block->handler_type != nullptr && "Catch block missing handler type");
        }
    }

    assert(found_try_block && "No try block found");
    assert(found_catch_block && "No catch block found");

    // Cleanup
    destroy_dex(context);
    std::cout << "Exception and basic block analysis tests passed successfully!" << std::endl;
    return 0;
}