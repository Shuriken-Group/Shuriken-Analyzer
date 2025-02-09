#include "dex-files-folder.inc"
#include "shuriken/api/C/shuriken_core.h"
#include <cassert>
#include <cstring>
#include <iostream>

const char *file = DEX_FILES_FOLDER
        "basic_test.dex";

int main() {
    hDexContext context = parse_dex(file);
    assert(context != nullptr && "Failed to parse DEX file");

    // Test string retrieval
    [[maybe_unused]] size_t str_count = get_number_of_strings(context);
    assert(str_count > 0 && "No strings found in DEX");

    [[maybe_unused]] const char *first_string = get_string_by_id(context, 0);
    assert(first_string != nullptr && "Failed to get first string");

    // Test class count
    [[maybe_unused]] uint16_t class_count = get_number_of_classes(context);
    assert(class_count > 0 && "No classes found in DEX");

    // Test class retrieval by id
    [[maybe_unused]] hdvmclass_t *first_class = get_class_by_id(context, 0);
    assert(first_class != nullptr && "Failed to get first class");
    assert(first_class->class_name != nullptr && "Class name is null");

    // Test class retrieval by name
    [[maybe_unused]] hdvmclass_t *basic_class = get_class_by_name(context, "BasicTest");
    assert(basic_class != nullptr && "Failed to get BasicTest class");

    // Test invalid cases
    assert(get_class_by_id(context, UINT16_MAX) == nullptr && "Invalid class ID should return null");
    assert(get_class_by_name(context, "NonExistentClass") == nullptr && "Non-existent class should return null");
    assert(get_string_by_id(context, SIZE_MAX) == nullptr && "Invalid string ID should return null");

    // Cleanup
    destroy_dex(context);
    std::cout << "Basic parsing tests passed successfully!" << std::endl;
    return 0;
}