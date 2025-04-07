#include <gtest/gtest.h>
#include "dex-files-folder.inc"
#include <shuriken/sdk/dex/dex.hpp>

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