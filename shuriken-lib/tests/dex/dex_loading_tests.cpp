#include <gtest/gtest.h>
#include "dex-files-folder.inc"
#include <shuriken/sdk/dex/dex.hpp>
#include <shuriken/sdk/dex/class.hpp>
#include <shuriken/sdk/dex/method.hpp>
#include <shuriken/sdk/dex/field.hpp>
#include <shuriken/sdk/dex/dvm_prototypes.hpp>
#include <shuriken/sdk/dex/dvm_types.hpp>

class DexTest : public ::testing::Test {
protected:
};

TEST_F(DexTest, LoadDexTest) {
    auto dex_file = shuriken::dex::Dex::create_from_file(std::string(DEX_FILES_FOLDER) + "/_int.dex");

    EXPECT_TRUE(dex_file.has_value());
    EXPECT_NE(dex_file.value(), nullptr);
}

TEST_F(DexTest, CreateFromFileFailure) {
    auto result = shuriken::dex::Dex::create_from_file("path/to/invalid/file");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().get_code(), shuriken::error::ErrorCode::FileNotFound);
    // or whatever error code you expect
}

TEST_F(DexTest, CheckClass) {
    auto dex_file = shuriken::dex::Dex::create_from_file(std::string(DEX_FILES_FOLDER) + "/_int.dex");

    EXPECT_TRUE(dex_file.has_value());
    EXPECT_NE(dex_file.value(), nullptr);

    const auto & dex = dex_file.value();
    const auto & cls = *(dex->get_classes().begin());
    EXPECT_EQ(cls.get_name_string(), "_int");
    EXPECT_EQ(cls.get_package_name_string(), "com/dexbox");
    EXPECT_EQ(cls.get_dalvik_name_string(), "Lcom/dexbox/_int;");
    EXPECT_EQ(cls.get_canonical_name_string(), "com.dexbox._int");
}

TEST_F(DexTest, CheckMethods) {
    auto dex_file = shuriken::dex::Dex::create_from_file(std::string(DEX_FILES_FOLDER) + "/_int.dex");

    EXPECT_TRUE(dex_file.has_value());
    EXPECT_NE(dex_file.value(), nullptr);

    const auto & dex = dex_file.value();
    auto & cls = *(dex->get_classes().begin());

    // Define expected method details
    struct MethodInfo {
        std::string name;
        std::string prototype;
        std::string class_name;
        std::string dex_name;
    };

    std::vector<MethodInfo> expected_methods = {
            {"<init>", "()V", "_int", "_int.dex"},
            {"main", "()I", "_int", "_int.dex"}
    };

    auto methods = cls.get_methods();
    ASSERT_EQ(cls.get_number_of_methods(), expected_methods.size());

    size_t i = 0;
    for (auto & method : cls.get_methods()) {
        const auto& expected = expected_methods[i++];

        EXPECT_EQ(method.get_name(), expected.name);
        EXPECT_EQ(method.get_method_prototype().get_descriptor(), expected.prototype);
        EXPECT_EQ(method.get_owner_class().get_name(), expected.class_name);
        EXPECT_EQ(method.get_owner_dex().get_dex_name(), expected.dex_name);
    }
}