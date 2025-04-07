#include <gtest/gtest.h>
#include <shuriken/sdk/dex/dvm_types.hpp>
#include <shuriken/sdk/dex/dvm_prototypes.hpp>
#include <shuriken/internal/providers/dex/dvm_types_provider.hpp>
#include <shuriken/internal/providers/dex/dvm_prototypes_provider.hpp>
#include <memory>
#include <vector>

using namespace shuriken::dex;
using namespace shuriken::dex::types;

class DVMPrototypeTest : public ::testing::Test {
protected:
    // Type providers
    DVMFundamentalProvider int_provider{"I", fundamental_e::INT};
    DVMFundamentalProvider void_provider{"V", fundamental_e::VOID};
    DVMFundamentalProvider bool_provider{"Z", fundamental_e::BOOLEAN};
    DVMClassProvider string_provider{"Ljava/lang/String;"};

    // Storage for our types
    std::vector<std::unique_ptr<DVMType>> type_storage;

    // Storage for our prototype objects
    std::vector<std::unique_ptr<DVMPrototypeProvider>> prototype_provider_storage;
    std::vector<std::unique_ptr<DVMPrototype>> prototype_storage;

    // Helper method to create and store a type
    DVMType &create_and_store_type(auto &provider) {
        auto type_ptr = std::make_unique<DVMType>(provider);
        type_storage.push_back(std::move(type_ptr));
        return *type_storage.back();
    }

    void SetUp() override {
        // Create basic types we'll need for testing
        create_and_store_type(int_provider);      // int
        create_and_store_type(void_provider);     // void
        create_and_store_type(bool_provider);     // boolean
        create_and_store_type(string_provider);   // String
    }

    // Helpers to get references to our stored types
    DVMType &get_int_type() { return *type_storage[0]; }
    DVMType &get_void_type() { return *type_storage[1]; }
    DVMType &get_bool_type() { return *type_storage[2]; }
    DVMType &get_string_type() { return *type_storage[3]; }
};

// Test a simple void method with no parameters
TEST_F(DVMPrototypeTest, VoidMethodNoParams) {
    // Create the prototype: void methodName()
    std::string shorty = "V";
    std::vector<dvmtype_t> params;

    auto provider = std::make_unique<DVMPrototypeProvider>(shorty, get_void_type(), params);
    auto prototype = std::make_unique<DVMPrototype>(*provider);

    // Store for ownership
    prototype_provider_storage.push_back(std::move(provider));
    prototype_storage.push_back(std::move(prototype));

    DVMPrototype& proto = *prototype_storage.back();

    // Test properties
    EXPECT_EQ(proto.get_shorty_idx(), "V");
    EXPECT_EQ(get_type(proto.get_return_type()), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(proto.get_return_type()), "V");

    // Test parameters (should be empty) - using manual count instead of std::distance
    auto params_iter = proto.get_parameters();

    // Alternative way to count elements
    int count = 0;

    for (const auto & it : params_iter) {
        count++;
    }
    EXPECT_EQ(count, 0);
}


// Test a method with primitive return type and parameters
TEST_F(DVMPrototypeTest, PrimitiveReturnAndParams) {
    // Create the prototype: int methodName(boolean, int)
    std::string shorty = "IZI";

    std::vector<dvmtype_t> params;
    params.emplace_back(std::ref(get_bool_type()));
    params.emplace_back(std::ref(get_int_type()));

    auto provider = std::make_unique<DVMPrototypeProvider>(shorty, get_int_type(), params);
    auto prototype = std::make_unique<DVMPrototype>(*provider);

    // Store for ownership
    prototype_provider_storage.push_back(std::move(provider));
    prototype_storage.push_back(std::move(prototype));

    DVMPrototype& proto = *prototype_storage.back();

    // Test properties
    EXPECT_EQ(proto.get_shorty_idx(), "IZI");
    EXPECT_EQ(get_type(proto.get_return_type()), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(proto.get_return_type()), "I");

    // Test parameters
    auto params_iter = proto.get_parameters();
    ASSERT_EQ(std::distance(params_iter.begin(), params_iter.end()), 2);

    auto it = params_iter.begin();
    EXPECT_EQ(get_type(*it), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(*it), "Z");

    ++it;
    EXPECT_EQ(get_type(*it), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(*it), "I");
}

// Test a method with object return type and parameters
TEST_F(DVMPrototypeTest, ObjectReturnAndParams) {
    // Create the prototype: String methodName(String, boolean)
    std::string shorty = "LLZ";

    std::vector<dvmtype_t> params;
    params.emplace_back(std::ref(get_string_type()));
    params.emplace_back(std::ref(get_bool_type()));

    auto provider = std::make_unique<DVMPrototypeProvider>(shorty, get_string_type(), params);
    auto prototype = std::make_unique<DVMPrototype>(*provider);

    // Store for ownership
    prototype_provider_storage.push_back(std::move(provider));
    prototype_storage.push_back(std::move(prototype));

    DVMPrototype& proto = *prototype_storage.back();

    // Test properties
    EXPECT_EQ(proto.get_shorty_idx(), "LLZ");
    EXPECT_EQ(get_type(proto.get_return_type()), type_e::CLASS);
    EXPECT_EQ(get_dalvik_format_string(proto.get_return_type()), "Ljava/lang/String;");
    EXPECT_EQ(get_canonical_name(proto.get_return_type()), "java.lang.String");

    // Test parameters
    auto params_iter = proto.get_parameters();
    ASSERT_EQ(std::distance(params_iter.begin(), params_iter.end()), 2);

    auto it = params_iter.begin();
    EXPECT_EQ(get_type(*it), type_e::CLASS);
    EXPECT_EQ(get_canonical_name(*it), "java.lang.String");

    ++it;
    EXPECT_EQ(get_type(*it), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(*it), "Z");
}

// Test a method with array parameters
TEST_F(DVMPrototypeTest, ArrayParameters) {
    // Create a new provider for the array to own - using new
    DVMTypeProvider* int_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMFundamentalProvider>,
                                                            "I", fundamental_e::INT);

    // Create the array provider with the raw pointer
    DVMArrayProvider array_provider{"[I", 1, int_provider_ptr};

    // Create and store the array type
    auto array_type_ptr = std::make_unique<DVMType>(array_provider);
    DVMType& int_array_type = *array_type_ptr;
    type_storage.push_back(std::move(array_type_ptr));

    // Create the prototype: void methodName(int[], String)
    std::string shorty = "V[L";

    std::vector<dvmtype_t> params;
    params.emplace_back(std::ref(int_array_type));
    params.emplace_back(std::ref(get_string_type()));

    auto provider = std::make_unique<DVMPrototypeProvider>(shorty, get_void_type(), params);
    auto prototype = std::make_unique<DVMPrototype>(*provider);

    // Store for ownership
    prototype_provider_storage.push_back(std::move(provider));
    prototype_storage.push_back(std::move(prototype));

    DVMPrototype& proto = *prototype_storage.back();

    // Test properties
    EXPECT_EQ(proto.get_shorty_idx(), "V[L");
    EXPECT_EQ(get_type(proto.get_return_type()), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(proto.get_return_type()), "V");

    // Test parameters
    auto params_iter = proto.get_parameters();
    ASSERT_EQ(std::distance(params_iter.begin(), params_iter.end()), 2);

    auto it = params_iter.begin();
    EXPECT_EQ(get_type(*it), type_e::ARRAY);
    EXPECT_EQ(get_dalvik_format_string(*it), "[I");
    EXPECT_EQ(get_canonical_name(*it), "int[]");

    ++it;
    EXPECT_EQ(get_type(*it), type_e::CLASS);
    EXPECT_EQ(get_canonical_name(*it), "java.lang.String");
}

// Test modification of prototypes through non-const references
TEST_F(DVMPrototypeTest, ModifyPrototype) {
    // Create initial prototype: void methodName()
    std::string shorty = "V";
    std::vector<dvmtype_t> params;

    auto provider = std::make_unique<DVMPrototypeProvider>(shorty, get_void_type(), params);
    auto prototype = std::make_unique<DVMPrototype>(*provider);

    // Store for ownership
    prototype_provider_storage.push_back(std::move(provider));
    prototype_storage.push_back(std::move(prototype));

    DVMPrototype& proto = *prototype_storage.back();

    // Initial validation
    EXPECT_EQ(get_dalvik_format_string(proto.get_return_type()), "V");

    // Now modify the return type reference
    // Note: This assumes the non-const get_return_type() returns a reference
    // that can be modified to point to a different type
    DVMType& return_type_ref = proto.get_return_type();

    // This is a tricky part - we can't directly assign a new DVMType because the
    // reference is to a specific variant instance. We'd need to modify the instance
    // it points to. This test checks we can access the non-const version.
    EXPECT_EQ(get_type(return_type_ref), type_e::FUNDAMENTAL);
    EXPECT_EQ(get_dalvik_format_string(return_type_ref), "V");
}

// Test the deref_iterator_range functionality with prototype parameters
TEST_F(DVMPrototypeTest, DerefIteratorRange) {
    // Create array type for testing - using new for the provider
    DVMTypeProvider* int_provider_ptr = new DVMTypeProvider(std::in_place_type<DVMFundamentalProvider>,
                                                            "I", fundamental_e::INT);

    // Create the array provider with the raw pointer
    DVMArrayProvider array_provider{"[I", 1, int_provider_ptr};

    // Create and store the array type
    auto array_type_ptr = std::make_unique<DVMType>(array_provider);
    DVMType& int_array_type = *array_type_ptr;
    type_storage.push_back(std::move(array_type_ptr));

    // Create a prototype with multiple parameters
    std::string shorty = "VIZL[I";

    // Set up parameters: boolean, int, String, int[]
    std::vector<dvmtype_t> params;
    params.emplace_back(std::ref(get_bool_type()));
    params.emplace_back(std::ref(get_int_type()));
    params.emplace_back(std::ref(get_string_type()));
    params.emplace_back(std::ref(int_array_type));

    auto provider = std::make_unique<DVMPrototypeProvider>(shorty, get_void_type(), params);
    auto prototype = std::make_unique<DVMPrototype>(*provider);

    // Store for ownership
    prototype_provider_storage.push_back(std::move(provider));
    prototype_storage.push_back(std::move(prototype));

    DVMPrototype& proto = *prototype_storage.back();

    // Get the parameters through the deref_iterator_range
    auto params_range = proto.get_parameters();

    // Test range-based for loop with the deref_iterator_range
    std::vector<types::type_e> expected_types = {
            type_e::FUNDAMENTAL,  // boolean
            type_e::FUNDAMENTAL,  // int
            type_e::CLASS,        // String
            type_e::ARRAY         // int[]
    };

    std::vector<types::type_e> actual_types;

    // Using range-based for loop with the deref_iterator_range
    for (const DVMType& type : params_range) {
        actual_types.push_back(get_type(type));
    }

    EXPECT_EQ(actual_types, expected_types);

    // Test iterator operations explicitly
    auto it = params_range.begin();
    EXPECT_EQ(get_dalvik_format_string(*it), "Z");  // boolean

    ++it;
    EXPECT_EQ(get_dalvik_format_string(*it), "I");  // int

    ++it;
    EXPECT_EQ(get_canonical_name(*it), "java.lang.String");  // String

    ++it;
    EXPECT_EQ(get_canonical_name(*it), "int[]");  // int[]

    ++it;
    EXPECT_EQ(it, params_range.end());

    // Test decrement operation
    --it;
    EXPECT_EQ(get_canonical_name(*it), "int[]");
}