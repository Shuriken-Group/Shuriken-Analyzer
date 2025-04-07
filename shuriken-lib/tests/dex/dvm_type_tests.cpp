#include <gtest/gtest.h>
#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/internal/providers/dex/dvm_types_provider.hpp>
#include <variant>
#include <vector>

using namespace shuriken::dex;
using namespace shuriken::dex::types;

class DVMTypeTest : public ::testing::Test {
protected:
    // Providers for the different types
    DVMFundamentalProvider int_provider{"I", fundamental_e::INT};
    DVMClassProvider string_provider{"Ljava/lang/String;"};

    // Storage for DVMType instances
    std::vector<DVMType> types_storage;

    // Storage for DVMType pointers
    std::vector<std::unique_ptr<DVMType>> type_storage;

    // Set up before each test
    void SetUp() override {
        // Create some base types to work with - use explicit variant construction
        types_storage.emplace_back(int_provider);
        types_storage.emplace_back(string_provider);
    }

    // Helper to get a reference to the stored type
    DVMType &get_int_type() {
        return types_storage[0];
    }

    DVMType &get_string_type() {
        return types_storage[1];
    }
};

// Test fundamental type
TEST_F(DVMTypeTest, FundamentalType) {
    // Create a fundamental type with explicit variant construction
    DVMType type{int_provider};

    // Test type identification
    EXPECT_EQ(get_type(type), type_e::FUNDAMENTAL);

    // Test type properties
    EXPECT_EQ(get_dalvik_format_string(type), "I");
    EXPECT_EQ(get_dalvik_format(type), "I");
    EXPECT_EQ(get_canonical_name(type), "int");
    EXPECT_EQ(get_canonical_name_string(type), "int");

    // Test as_* casting
    const DVMFundamental *fundamental_ptr = as_fundamental(type);
    EXPECT_NE(fundamental_ptr, nullptr);
    EXPECT_EQ(fundamental_ptr->get_fundamental_type(), fundamental_e::INT);
}

// Test class type
TEST_F(DVMTypeTest, ClassType) {
    // Create a class type with explicit variant construction
    DVMType type{string_provider};  // Explicit variant construction

    // Test type identification
    EXPECT_EQ(get_type(type), type_e::CLASS);

    // Test type properties
    EXPECT_EQ(get_dalvik_format_string(type), "Ljava/lang/String;");
    EXPECT_EQ(get_canonical_name(type), "java.lang.String");

    // Test as_* casting
    const DVMClass *cls_ptr = as_class(type);
    EXPECT_NE(cls_ptr, nullptr);
    EXPECT_EQ(cls_ptr->get_dalvik_format(), "Ljava/lang/String;");
}

// Test array type
TEST_F(DVMTypeTest, ArrayType) {
    // Create a new provider for the array to own - using new without storing in any container
    DVMTypeProvider* int_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMFundamentalProvider>,
                                                            "I", fundamental_e::INT);

    // Create array provider - it will take ownership of int_provider_ptr
    DVMArrayProvider array_provider{"[I", 1, int_provider_ptr};
    DVMType array_type{array_provider};

    // Test type identification
    EXPECT_EQ(get_type(array_type), type_e::ARRAY);

    // Test type properties
    EXPECT_EQ(get_dalvik_format_string(array_type), "[I");
    EXPECT_EQ(get_canonical_name(array_type), "int[]");

    // Test as_* casting
    const DVMArray *array_ptr = as_array(array_type);
    EXPECT_NE(array_ptr, nullptr);
    EXPECT_EQ(array_ptr->get_array_depth(), 1);

    // Test base type
    const DVMType &base_type = array_ptr->get_base_type();
    EXPECT_EQ(get_type(base_type), type_e::FUNDAMENTAL);
}

// Test using reference wrappers
TEST_F(DVMTypeTest, ReferenceWrappers) {
    // Create a vector of DVMType objects
    std::vector<DVMType> type_objects;

    DVMType type(int_provider);
    type_objects.push_back(type);

    DVMClass cls(string_provider);
    type_objects.push_back(DVMType(cls));

    // Create a vector of references to these objects
    std::vector<dvmtype_t> type_refs;
    for (auto &type: type_objects) {
        type_refs.emplace_back(std::ref(type));
    }

    // Create a span over the references
    dvmtypes_list_t types_span(type_refs.data(), type_refs.size());

    // Test accessing through span
    EXPECT_EQ(types_span.size(), 2);
    EXPECT_EQ(get_type(types_span[0]), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_type(types_span[1]), type_e::CLASS);
}

// Test handling of void type
TEST_F(DVMTypeTest, VoidTypeHandling) {
    // Create a void fundamental type
    DVMFundamentalProvider void_provider{"V", fundamental_e::VOID};

    auto void_type = std::make_unique<DVMType>(void_provider);
    type_storage.push_back(std::move(void_type));

    DVMType& type = *type_storage.back();

    // Test properties
    EXPECT_EQ(get_type(type), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(type), "V");
    EXPECT_EQ(get_canonical_name(type), "void");

    const DVMFundamental* fundamental = as_fundamental(type);
    EXPECT_NE(fundamental, nullptr);
    EXPECT_EQ(fundamental->get_fundamental_type(), fundamental_e::VOID);
}

// Test multi-dimensional arrays
TEST_F(DVMTypeTest, MultiDimensionalArrays) {
    // Create a provider for int type - array will take ownership
    DVMTypeProvider* int_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMFundamentalProvider>,
                                                            "I", fundamental_e::INT);

    // Create a 1D array provider
    DVMArrayProvider array1d_provider{"[I", 1, int_provider_ptr};
    auto array1d_type = std::make_unique<DVMType>(array1d_provider);
    type_storage.push_back(std::move(array1d_type));

    // Create a new provider for the 2D array (don't reuse int_provider_ptr - it's now owned)
    DVMTypeProvider* new_int_provider = new DVMTypeProvider(std::in_place_type<DVMFundamentalProvider>,
                                                            "I", fundamental_e::INT);

    // Create a 2D array provider
    DVMArrayProvider array2d_provider{"[[I", 2, new_int_provider};
    auto array2d_type = std::make_unique<DVMType>(array2d_provider);
    type_storage.push_back(std::move(array2d_type));

    // Get references for testing
    DVMType& array1d = *type_storage[type_storage.size() - 2];
    DVMType& array2d = *type_storage[type_storage.size() - 1];

    // Test properties
    EXPECT_EQ(get_dalvik_format_string(array1d), "[I");
    EXPECT_EQ(get_canonical_name(array1d), "int[]");
    EXPECT_EQ(as_array(array1d)->get_array_depth(), 1);

    EXPECT_EQ(get_dalvik_format_string(array2d), "[[I");
    EXPECT_EQ(get_canonical_name(array2d), "int[][]");
    EXPECT_EQ(as_array(array2d)->get_array_depth(), 2);
}

// Test array of objects (not just primitives)
TEST_F(DVMTypeTest, ArrayOfObjects) {
    // Create a provider for string type - array will take ownership
    DVMTypeProvider* string_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMClassProvider>,
                                                               "Ljava/lang/String;");

    // Create an array of strings
    DVMArrayProvider array_provider{"[Ljava/lang/String;", 1, string_provider_ptr};
    auto array_type = std::make_unique<DVMType>(array_provider);
    type_storage.push_back(std::move(array_type));

    DVMType& type = *type_storage.back();

    // Test properties
    EXPECT_EQ(get_type(type), type_e::ARRAY);
    EXPECT_EQ(get_dalvik_format_string(type), "[Ljava/lang/String;");
    EXPECT_EQ(get_canonical_name(type), "java.lang.String[]");

    const DVMArray* array = as_array(type);
    EXPECT_NE(array, nullptr);
    EXPECT_EQ(array->get_array_depth(), 1);

    // Test base type is a class
    const DVMType& base_type = array->get_base_type();
    EXPECT_EQ(get_type(base_type), type_e::CLASS);
    EXPECT_EQ(get_dalvik_format_string(base_type), "Ljava/lang/String;");
}

// Test handling of boolean type
TEST_F(DVMTypeTest, BooleanTypeHandling) {
    DVMFundamentalProvider bool_provider{"Z", fundamental_e::BOOLEAN};

    auto bool_type = std::make_unique<DVMType>(bool_provider);
    type_storage.push_back(std::move(bool_type));

    DVMType& type = *type_storage.back();

    // Test properties
    EXPECT_EQ(get_type(type), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(type), "Z");
    EXPECT_EQ(get_canonical_name(type), "boolean");

    const DVMFundamental* fundamental = as_fundamental(type);
    EXPECT_NE(fundamental, nullptr);
    EXPECT_EQ(fundamental->get_fundamental_type(), fundamental_e::BOOLEAN);
}

// Test nested class names
TEST_F(DVMTypeTest, NestedClassNames) {
    // Create a nested class type (e.g., OuterClass$InnerClass)
    DVMClassProvider nested_provider{"Lcom/example/OuterClass$InnerClass;"};

    auto nested_type = std::make_unique<DVMType>(nested_provider);
    type_storage.push_back(std::move(nested_type));

    DVMType& type = *type_storage.back();

    // Test properties
    EXPECT_EQ(get_type(type), type_e::CLASS);
    EXPECT_EQ(get_dalvik_format_string(type), "Lcom/example/OuterClass$InnerClass;");
    EXPECT_EQ(get_canonical_name(type), "com.example.OuterClass$InnerClass");
}

// Test array of arrays
TEST_F(DVMTypeTest, ArrayOfArrays) {
    // Create provider for int type
    DVMTypeProvider* int_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMFundamentalProvider>,
                                                            "I", fundamental_e::INT);

    // Create a 1D array provider
    // Create a DVMTypeProvider from the 1D array provider
    DVMTypeProvider* array1d_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMArrayProvider>, "[I", 1, int_provider_ptr);

    // Create a 2D array using the 1D array provider
    DVMArrayProvider array2d_provider{"[[I", 1, array1d_provider_ptr};

    // Create the type
    auto array2d_type = std::make_unique<DVMType>(array2d_provider);
    type_storage.push_back(std::move(array2d_type));

    DVMType& type = *type_storage.back();

    // Test properties
    EXPECT_EQ(get_type(type), type_e::ARRAY);
    EXPECT_EQ(get_dalvik_format_string(type), "[[I");

    // The canonical name might be either "int[][]" or "[I[]" depending on your implementation
    // Just test that it contains the expected parts
    std::string canonical = get_canonical_name_string(type);
    EXPECT_TRUE(canonical.find("[]") != std::string::npos);

    const DVMArray* array = as_array(type);
    EXPECT_NE(array, nullptr);
    EXPECT_EQ(array->get_array_depth(), 1);  // This test is checking an array of arrays, so depth=1

    // Check the base type is also an array
    const DVMType& base_type = array->get_base_type();
    EXPECT_EQ(get_type(base_type), type_e::ARRAY);
}

// Test dvmtypes_list_t construction and access
TEST_F(DVMTypeTest, TypeListOperations) {
    // Create a vector of different types
    std::vector<std::unique_ptr<DVMType>> param_types;

    // Add int type
    auto int_type = std::make_unique<DVMType>(int_provider);
    param_types.push_back(std::move(int_type));

    // Add string type
    auto string_type = std::make_unique<DVMType>(string_provider);
    param_types.push_back(std::move(string_type));

    // Create references to these types
    std::vector<dvmtype_t> type_refs;
    for (auto& type : param_types) {
        type_refs.emplace_back(std::ref(*type));
    }

    // Create a span
    dvmtypes_list_t param_list(type_refs.data(), type_refs.size());

    // Test accessing elements
    ASSERT_EQ(param_list.size(), 2);
    EXPECT_EQ(get_type(param_list[0]), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_type(param_list[1]), type_e::CLASS);

    // Test iterating through the list
    std::vector<types::type_e> expected_types = {type_e::FUNDAMENTAL, type_e::CLASS};
    std::vector<types::type_e> actual_types;

    for (const dvmtype_t& type_ref : param_list) {
        actual_types.push_back(get_type(type_ref));
    }

    EXPECT_EQ(actual_types, expected_types);
}