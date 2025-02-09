// test_methods.cpp
#include <cassert>
#include <cstring>
#include <iostream>
#include "dex-files-folder.inc"
#include "shuriken/api/C/shuriken_core.h"

const char *file = DEX_FILES_FOLDER
    "method_test.dex";

int main() {
    hDexContext context = parse_dex(file);
    assert(context != nullptr && "Failed to parse DEX file");

    // Get the test class
    hdvmclass_t* test_class = get_class_by_name(context, "MethodTest");
    assert(test_class != nullptr && "Failed to get MethodTest class");

    // Test direct methods (constructor and private methods)
    assert(test_class->direct_methods_size > 0 && "No direct methods found");
    [[maybe_unused]] bool found_constructor = false;
    [[maybe_unused]] bool found_private_method = false;

    for (uint16_t i = 0; i < test_class->direct_methods_size; i++) {
        hdvmmethod_t* method = &test_class->direct_methods[i];
        
        if (strcmp(method->method_name, "<init>") == 0) {
            found_constructor = true;
        }
        
        if (strcmp(method->method_name, "privateMethod") == 0) {
            found_private_method = true;
            assert((method->access_flags & ACC_PRIVATE) && "Private method not marked private");
        }
    }

    assert(found_constructor && "Constructor not found");
    assert(found_private_method && "Private method not found");

    // Test virtual methods (public instance methods)
    assert(test_class->virtual_methods_size > 0 && "No virtual methods found");
    [[maybe_unused]] bool found_public_method = false;

    for (uint16_t i = 0; i < test_class->virtual_methods_size; i++) {
        hdvmmethod_t* method = &test_class->virtual_methods[i];
        
        if (strcmp(method->method_name, "publicMethod") == 0) {
            found_public_method = true;
            assert((method->access_flags & ACC_PUBLIC) && "Public method not marked public");
            
            // Test method specific information
            assert(method->prototype != nullptr && "Method prototype is null");
            assert(method->class_name != nullptr && "Method class name is null");
        }
    }

    assert(found_public_method && "Public method not found");

    // Test method disassembly
    disassemble_dex(context);
    
    // Get a specific method by name
    hdvmmethod_t* test_method = get_method_by_name(context, "LMethodTest;->publicMethod()V");
    assert(test_method != nullptr && "Failed to get test method");
    
    // Get disassembled method
    [[maybe_unused]] dvmdisassembled_method_t* disassembled = get_disassembled_method(context, test_method->dalvik_name);
    assert(disassembled != nullptr && "Failed to get disassembled method");
    assert(disassembled->n_of_instructions > 0 && "No instructions in disassembled method");
    assert(disassembled->method_string != nullptr && "No method string available");

    // Cleanup
    destroy_dex(context);
    std::cout << "Method tests passed successfully!" << std::endl;
    return 0;
}