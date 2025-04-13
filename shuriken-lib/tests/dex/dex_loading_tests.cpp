#include <gtest/gtest.h>
#include "dex-files-folder.inc"
#include <shuriken/sdk/dex/dex.hpp>
#include <shuriken/sdk/dex/class.hpp>

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