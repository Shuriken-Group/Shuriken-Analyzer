#include <cassert>
#include <cstring>
#include <iostream>
#include "dex-files-folder.inc"
#include "shuriken/api/C/shuriken_core.h"

const char *file = DEX_FILES_FOLDER
    "field_test.dex";

int main() {
    hDexContext context = parse_dex(file);
    assert(context != nullptr && "Failed to parse DEX file");

    // Get the test class
    hdvmclass_t* test_class = get_class_by_name(context, "FieldTest");
    assert(test_class != nullptr && "Failed to get FieldTest class");

    // Test instance fields
    assert(test_class->instance_fields_size > 0 && "No instance fields found");
    [[maybe_unused]] bool found_private_field = false;
    [[maybe_unused]] bool found_public_field = false;

    for (uint16_t i = 0; i < test_class->instance_fields_size; i++) {
        hdvmfield_t* field = &test_class->instance_fields[i];
        
        if (strcmp(field->name, "privateField") == 0) {
            found_private_field = true;
            assert((field->access_flags & ACC_PRIVATE) && "Private field not marked private");
        }
        
        if (strcmp(field->name, "publicField") == 0) {
            found_public_field = true;
            assert((field->access_flags & ACC_PUBLIC) && "Public field not marked public");
        }
    }

    assert(found_private_field && "Private field not found");
    assert(found_public_field && "Public field not found");

    // Test static fields
    assert(test_class->static_fields_size > 0 && "No static fields found");
    [[maybe_unused]] bool found_static_field = false;

    for (uint16_t i = 0; i < test_class->static_fields_size; i++) {
        hdvmfield_t* field = &test_class->static_fields[i];
        
        if (strcmp(field->name, "staticField") == 0) {
            found_static_field = true;
            assert((field->access_flags & ACC_STATIC) && "Static field not marked static");
        }
    }

    assert(found_static_field && "Static field not found");

    // Cleanup
    destroy_dex(context);
    std::cout << "Field tests passed successfully!" << std::endl;
    return 0;
}