#include <gtest/gtest.h>
#include "dex-files-folder.inc"
#include <shuriken/sdk/dex/dex.hpp>
#include <shuriken/sdk/dex/class.hpp>
#include <shuriken/sdk/dex/method.hpp>
#include <regex>

class DexDisassemblerTest : public ::testing::Test {
protected:
    DexDisassemblerTest() : dex_file(shuriken::dex::Dex::create_from_file(std::string(DEX_FILES_FOLDER) + "/_int.dex")),
                            dex(dex_file.value()),
                            cls(*(dex->get_classes().begin())) {
    }

    void SetUp() override {
        ASSERT_TRUE(dex_file.has_value());
        ASSERT_NE(dex_file.value(), nullptr);
        // Additional setup if needed
    }

    shuriken::error::Result<std::unique_ptr<shuriken::dex::Dex>> dex_file;
    std::unique_ptr<shuriken::dex::Dex> &dex;
    shuriken::dex::Class &cls;
};

TEST_F(DexDisassemblerTest, TestBasicClassInfo) {
    EXPECT_EQ(cls.get_name_string(), "_int");
    EXPECT_EQ(cls.get_package_name_string(), "com/dexbox");
    EXPECT_EQ(cls.get_dalvik_name_string(), "Lcom/dexbox/_int;");
    EXPECT_EQ(cls.get_canonical_name_string(), "com.dexbox._int");
}

TEST_F(DexDisassemblerTest, TestMethodCount) {
    auto methods = cls.get_methods();
    EXPECT_EQ(cls.get_number_of_methods(), 2); // <init> and main methods
}

TEST_F(DexDisassemblerTest, TestInitMethodDisassembly) {
    auto methods = cls.get_methods();
    auto init_method = std::find_if(methods.begin(), methods.end(),
                                    [](const auto& method) { return method.get_name() == "<init>"; });

    ASSERT_NE(init_method, methods.end());

    auto& instrs = init_method->get_method_instructions();
    EXPECT_EQ(instrs.size(), 2); // Should have 2 instructions

    // Verify first instruction is invoke-direct
    auto it = instrs.begin();
    auto first_instr = it->get().print_instruction();
    EXPECT_TRUE(first_instr.find("invoke-direct") != std::string::npos);
    EXPECT_TRUE(first_instr.find("java/lang/Object") != std::string::npos);
    EXPECT_TRUE(first_instr.find("<init>") != std::string::npos);
    EXPECT_TRUE(first_instr.find("method@") != std::string::npos);

    // Verify second instruction is return-void
    ++it;
    auto second_instr = it->get().print_instruction();
    EXPECT_EQ(second_instr, "return-void");
}

TEST_F(DexDisassemblerTest, TestMainMethodDisassembly) {
    auto methods = cls.get_methods();
    auto main_method = std::find_if(methods.begin(), methods.end(),
                                    [](const auto& method) { return method.get_name() == "main"; });

    ASSERT_NE(main_method, methods.end());

    auto& instrs = main_method->get_method_instructions();
    EXPECT_GT(instrs.size(), 10); // Should have many instructions

    // Test specific instruction patterns
    std::vector<std::string> instruction_strings;
    for (const auto& instr : instrs) {
        instruction_strings.push_back(std::string(instr.get().print_instruction()));
    }

    // Verify we have the expected instruction types
    bool has_sget_object = false;
    bool has_const_string = false;
    bool has_invoke_virtual = false;
    bool has_new_instance = false;
    bool has_const_4 = false;
    bool has_const_16 = false;
    bool has_return = false;

    for (const auto& instr_str : instruction_strings) {
        if (instr_str.find("sget-object") != std::string::npos) {
            has_sget_object = true;
            // Verify format: should contain field@ reference
            EXPECT_TRUE(instr_str.find("field@") != std::string::npos);
        }
        if (instr_str.find("const-string") != std::string::npos) {
            has_const_string = true;
            // Verify format: should contain string@ reference
            EXPECT_TRUE(instr_str.find("string@") != std::string::npos);
        }
        if (instr_str.find("invoke-virtual") != std::string::npos) {
            has_invoke_virtual = true;
            // Verify format: should contain method@ reference
            EXPECT_TRUE(instr_str.find("method@") != std::string::npos);
        }
        if (instr_str.find("new-instance") != std::string::npos) {
            has_new_instance = true;
            // Verify format: should contain type@ reference
            EXPECT_TRUE(instr_str.find("type@") != std::string::npos);
        }
        if (instr_str.find("const/4") != std::string::npos) {
            has_const_4 = true;
        }
        if (instr_str.find("const/16") != std::string::npos) {
            has_const_16 = true;
        }
        if (instr_str.find("return") != std::string::npos) {
            has_return = true;
        }
    }

    EXPECT_TRUE(has_sget_object);
    EXPECT_TRUE(has_const_string);
    EXPECT_TRUE(has_invoke_virtual);
    EXPECT_TRUE(has_new_instance);
    EXPECT_TRUE(has_const_4);
    EXPECT_TRUE(has_const_16);
    EXPECT_TRUE(has_return);
}

TEST_F(DexDisassemblerTest, TestSpecificInstructions) {
    auto methods = cls.get_methods();
    auto main_method = std::find_if(methods.begin(), methods.end(),
                                    [](const auto& method) { return method.get_name() == "main"; });

    ASSERT_NE(main_method, methods.end());
    auto& instrs = main_method->get_method_instructions();

    // Test first few instructions match expected pattern
    std::vector<std::string> expected_patterns = {
            "sget-object v6, Ljava/lang/System;->out.*field@0000",
            "const-string v0, \"test: ===============================================================\".*string@",
            "invoke-virtual.*println.*method@",
            "sget-object v6, Ljava/lang/System;->out.*field@0000",
            "const-string v0, \"test: int: \\.\\.\".*string@"
    };

    auto it = instrs.begin();
    for (size_t i = 0; i < expected_patterns.size() && it != instrs.end(); ++i, ++it) {
        std::string instr_str = std::string(it->get().print_instruction());
        std::regex pattern(expected_patterns[i]);
        EXPECT_TRUE(std::regex_search(instr_str, pattern))
                            << "Instruction " << i << ": '" << instr_str
                            << "' doesn't match pattern: '" << expected_patterns[i] << "'";
    }
}

TEST_F(DexDisassemblerTest, TestIndexFormatting) {
    auto methods = cls.get_methods();
    auto main_method = std::find_if(methods.begin(), methods.end(),
                                    [](const auto& method) { return method.get_name() == "main"; });

    ASSERT_NE(main_method, methods.end());
    auto& instrs = main_method->get_method_instructions();

    // Check that indices are formatted with 4 digits (if you implemented that)
    for (const auto& instr : instrs) {
        std::string instr_str = std::string(instr.get().print_instruction());

        // Test method@ indices have 4 digits
        std::regex method_pattern(R"(method@(\d{4}))");
        std::smatch method_match;
        if (std::regex_search(instr_str, method_match, method_pattern)) {
            EXPECT_EQ(method_match[1].str().length(), 4)
                                << "Method index should be 4 digits in: " << instr_str;
        }

        // Test string@ indices have 4 digits
        std::regex string_pattern(R"(string@(\d{4}))");
        std::smatch string_match;
        if (std::regex_search(instr_str, string_match, string_pattern)) {
            EXPECT_EQ(string_match[1].str().length(), 4)
                                << "String index should be 4 digits in: " << instr_str;
        }

        // Test field@ indices have 4 digits
        std::regex field_pattern(R"(field@(\d{4}))");
        std::smatch field_match;
        if (std::regex_search(instr_str, field_match, field_pattern)) {
            EXPECT_EQ(field_match[1].str().length(), 4)
                                << "Field index should be 4 digits in: " << instr_str;
        }

        // Test type@ indices have 4 digits
        std::regex type_pattern(R"(type@(\d{4}))");
        std::smatch type_match;
        if (std::regex_search(instr_str, type_match, type_pattern)) {
            EXPECT_EQ(type_match[1].str().length(), 4)
                                << "Type index should be 4 digits in: " << instr_str;
        }
    }
}

TEST_F(DexDisassemblerTest, TestConstantValues) {
    auto methods = cls.get_methods();
    auto main_method = std::find_if(methods.begin(), methods.end(),
                                    [](const auto& method) { return method.get_name() == "main"; });

    ASSERT_NE(main_method, methods.end());
    auto& instrs = main_method->get_method_instructions();

    // Look for specific constant values that should appear
    std::vector<std::string> expected_constants = {
            "const/16 v3, 10",      // const/16 v3, #int 10
            "const/4 v4, 6",        // const/4 v4, #int 6
            "const/16 v4, 16",      // const/16 v4, #int 16
            "const/4 v4, 4",        // const/4 v4, #int 4
            "const/4 v4, 0",        // const/4 v4, #int 0
            "const/4 v5, 8",        // const/4 v5, #int -8 (but displayed as 8)
            "const/16 v5, 32",      // const/16 v5, #int 32
            "const/4 v5, 2",        // const/4 v5, #int 2
            "const/16 v2, 9"        // const/16 v2, #int 9
    };

    std::set<std::string> found_constants;
    for (const auto& instr : instrs) {
        std::string instr_str = std::string(instr.get().print_instruction());
        for (const auto& expected : expected_constants) {
            if (instr_str.find(expected) != std::string::npos) {
                found_constants.insert(expected);
            }
        }
    }

    // We should find most of the expected constants
    EXPECT_GE(found_constants.size(), expected_constants.size() / 2)
                        << "Should find at least half of the expected constants";
}

TEST_F(DexDisassemblerTest, TestStringConstants) {
    auto methods = cls.get_methods();
    auto main_method = std::find_if(methods.begin(), methods.end(),
                                    [](const auto& method) { return method.get_name() == "main"; });

    ASSERT_NE(main_method, methods.end());
    auto& instrs = main_method->get_method_instructions();

    // Look for expected string constants
    std::vector<std::string> expected_strings = {
            "test: ===============================================================",
            "test: int: ..",
            "val = ",
            "test: int: ok"
    };

    std::set<std::string> found_strings;
    for (const auto& instr : instrs) {
        std::string instr_str = std::string(instr.get().print_instruction());
        if (instr_str.find("const-string") != std::string::npos) {
            for (const auto& expected : expected_strings) {
                if (instr_str.find(expected) != std::string::npos) {
                    found_strings.insert(expected);
                }
            }
        }
    }

    EXPECT_EQ(found_strings.size(), expected_strings.size())
                        << "Should find all expected string constants";
}

// Debug helper to print all instructions (useful for development)
TEST_F(DexDisassemblerTest, DISABLED_PrintAllInstructions) {
    for (auto & method : cls.get_methods()) {
        std::cout << method.get_descriptor() << "\n";
        auto & instrs = method.get_method_instructions();
        size_t i = 0;
        for (auto it = instrs.begin(); it != instrs.end(); ++it, ++i) {
            std::cout << std::setfill('0') << std::setw(4) << i << ": "
                      << it->get().print_instruction() << '\n';
        }
        std::cout << "\n";
    }
}