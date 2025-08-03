//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <chrono>
#include <fmt/core.h>
#include <functional>
#include <iostream>
#include <vector>

#include <shuriken/sdk/dex/dex.hpp>
#include <shuriken/sdk/dex/class.hpp>
#include <shuriken/sdk/dex/method.hpp>
#include <shuriken/sdk/dex/field.hpp>
#include <shuriken/sdk/dex/instruction.hpp>
#include <shuriken/sdk/dex/disassembly_constants.hpp>
#include <shuriken/sdk/dex/control_flow_graph.hpp>

void show_help(std::string &prog_name) {
    fmt::println("USAGE: {} [-dex <dex_file_to_analyze>] [-h] [-m] [-b] [-D]", prog_name);
    fmt::println(" -dex <dex_file_to_analyze>: specify a dex file to analyze");
    fmt::println(" -c: show classes from file");
    fmt::println(" -f: show fields from classes (it needs -c)");
    fmt::println(" -m: show methods from classes (it needs -c)");
    fmt::println(" -b: show bytecode from methods (it needs -m)");
    fmt::println(" -D: show the disassembled code from methods (it needs -m)");
    fmt::println(" -G: show the disassembled code in graph mode (it needs -m)");
}

std::string dex_file_str;
bool show_classes = false;
bool methods = false;
bool fields = false;
bool code = false;
bool disassembly = false;
bool graph = false;

shuriken::error::Result<std::unique_ptr<shuriken::dex::Dex>> dex_file;

void parse_dex(std::string &dex_file_str);

void print_classes(shuriken::dex::Dex &);

void print_method(shuriken::dex::Method &);

void print_field(shuriken::dex::Field &);

void print_code(shuriken::dex::Method &);

void print_graph(shuriken::dex::Method &);

int main(int argc, char **argv) {
    std::vector<std::string> args{argv, argv + argc};

    auto start_time = std::chrono::high_resolution_clock::now();

    if (args.size() == 1) {
        show_help(args[0]);
        return -1;
    }

    std::unordered_map<std::string, std::function<void()>> options{
            {"-c", [&]() { show_classes = true; }},
            {"-m", [&]() { methods = true; }},
            {"-f", [&]() { fields = true; }},
            {"-b", [&]() { code = true; }},
            {"-D", [&]() { disassembly = true; }},
            {"-G", [&]() { graph = true; }}
    };

    std::unordered_map<std::string, std::function<void(std::string &)>> option_file{
            {"-dex", [&](std::string &param) {
                if (!param.ends_with(".dex")) {
                    fmt::println("ERROR file {} provided is not a dex file.", param);
                    std::exit(1);
                }
                dex_file_str = param;
            }}
    };

    for (auto it = args.begin(); it != args.end(); it++) {
        auto &param = *it;
        if (auto opt = options.find(param); opt != options.end())
            opt->second();
        if (auto opt = option_file.find(param); opt != option_file.end()) {
            it++;
            if (it == args.end()) {
                fmt::println("ERROR number of arguments provided mismatch");
                std::exit(1);
            }
            auto &str = *it;
            opt->second(str);
        }
    }

    if (!dex_file_str.empty()) {
        parse_dex(dex_file_str);

        if (show_classes) {
            std::unique_ptr<shuriken::dex::Dex> &dex = dex_file.value();
            print_classes(*dex);
        }
    }
}

void parse_dex(std::string &dex_file_str) {
    dex_file = shuriken::dex::Dex::create_from_file(dex_file_str);

    if (!dex_file.has_value()) {
        const auto &error = dex_file.error();

        fmt::println("ERROR generating the DEX object {}", error.get_message());

        std::exit(3);
    }
}

void print_classes(shuriken::dex::Dex &dex) {
    for (auto &cls: dex.get_classes()) {
        fmt::println("CLASS: {}", cls.get_name());
        fmt::println("CANONICAL NAME: {}", cls.get_canonical_name());
        fmt::println("DALVIK NAME: {}", cls.get_dalvik_name());

        if (methods) {
            size_t i = 0;
            for (auto &method: cls.get_methods()) {
                fmt::println("Method[{}]:", i++);
                print_method(method);
            }
        }
        if (fields) {
            size_t i = 0;
            for (auto &field: cls.get_fields()) {
                fmt::println("Field[{}]", i++);
                print_field(field);
            }
        }
    }
}

void print_method(shuriken::dex::Method &method) {
    fmt::println("\tMETHOD: {}", method.get_descriptor());
    fmt::println("\tACCESS FLAGS: 0x{:04X} ({})", static_cast<std::uint16_t>(method.get_method_access_flags()),
                 method.get_method_access_flags_str());
    fmt::print("\tMETHOD TYPE: ");
    if (method.get_method_type() == shuriken::dex::types::method_type_e::DIRECT_METHOD)
        fmt::println("DIRECT");
    else
        fmt::println("VIRTUAL");

    if (disassembly)
        print_code(method);
    if (graph)
        print_graph(method);
}

void print_field(shuriken::dex::Field &field) {
    fmt::println("\tFIELD: {}", field.get_descriptor());
    fmt::println("\tACCESS FLAGS: 0x{:04X} ({})", static_cast<std::uint16_t>(field.get_field_access_flags()),
                 field.get_field_access_flags_str());
    fmt::print("\tTYPE:");
    if (field.get_type() == shuriken::dex::types::field_type_e::INSTANCE_FIELD)
        fmt::println("INSTANCE");
    else
        fmt::println("STATIC");
}

std::string format_bytes_span(std::span<const std::uint8_t> data) {
    constexpr size_t MAX_PAIRS = 7;  // Maximum 7 pairs (14 bytes) before "..."
    constexpr size_t EXPECTED_LENGTH = 34;  // Length of "0001 0f00 0100 0000 7e00 0000 7600 ..."

    std::string result;
    result.reserve(EXPECTED_LENGTH);

    if (data.size() >= MAX_PAIRS * 2) {
        // We have enough bytes for all 7 pairs, show them + "..."
        for (size_t i = 0; i < MAX_PAIRS * 2; i += 2) {
            if (i > 0) result += " ";
            result += fmt::format("{:02x}{:02x}", data[i], data[i + 1]);
        }
        result += " ...";
    } else {
        // We have fewer than 14 bytes, show what we have and pad with spaces
        size_t byte_idx = 0;

        // Process complete pairs
        for (size_t pair = 0; pair < MAX_PAIRS; ++pair) {
            if (pair > 0) result += " ";

            if (byte_idx + 1 < data.size()) {
                // Complete pair available
                result += fmt::format("{:02x}{:02x}", data[byte_idx], data[byte_idx + 1]);
                byte_idx += 2;
            } else if (byte_idx < data.size()) {
                // Only one byte left, pad with zeros
                result += fmt::format("{:02x}00", data[byte_idx]);
                byte_idx += 1;
            } else {
                // No more bytes, fill with spaces
                result += "    ";  // 4 spaces for missing pair
            }
        }

        // Add trailing spaces to match exact length
        while (result.length() < EXPECTED_LENGTH - 4) {  // -4 for " ..."
            result += " ";
        }
        result += "    ";  // 4 spaces instead of " ..."
    }

    return result;
}

void print_code(shuriken::dex::Method &method) {
    fmt::println("\tNUMBER OF REGISTERS: {}", method.registers_size());
    fmt::println("\tCODE:");
    for (auto &instr_ref: method.get_method_instructions()) {
        auto &instr = instr_ref.get();
        fmt::println("{:08X} {} | {}", instr.get_address(),
                     format_bytes_span(instr.get_instruction_bytecode()),
                     instr.print_instruction());
    }
    for (auto &exception: method.get_exceptions()) {
        fmt::println("\tException try-start addr: 0x{:08X}", exception.try_value_start_addr);
        fmt::println("\tException try-end addr: 0x{:08X}", exception.try_value_end_addr);
        for (auto &catch_info: exception.handler) {
            fmt::println("\t\tCatch address: 0x{:08X}", catch_info.handler_start_addr);
            if (catch_info.handler_data)
                fmt::println("\t\tCaught exception: {}", shuriken::dex::get_canonical_name(*catch_info.handler_data));
        }
    }
}

void print_graph(shuriken::dex::Method &method) {
    auto & cfg = method.get_control_flow_graph();

    fmt::println("{}", cfg.toString());
}