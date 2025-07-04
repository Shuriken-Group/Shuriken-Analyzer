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

void show_help(std::string &prog_name) {
    fmt::println("USAGE: {} [-dex <dex_file_to_analyze>] [-h] [-m] [-b] [-D]", prog_name);
    fmt::println(" -dex <dex_file_to_analyze>: specify a dex file to analyze");
    fmt::println(" -c: show classes from file");
    fmt::println(" -f: show fields from classes (it needs -c)");
    fmt::println(" -m: show methods from classes (it needs -c)");
    fmt::println(" -b: show bytecode from methods (it needs -m)");
    fmt::println(" -D: show the disassembled code from methods (it needs -m)");
}

std::string dex_file_str;
bool show_classes = false;
bool methods = false;
bool fields = false;
bool code = false;
bool disassembly = false;

shuriken::error::Result<std::unique_ptr<shuriken::dex::Dex>> dex_file;

void parse_dex(std::string &dex_file_str);

void print_classes(shuriken::dex::Dex &);

void print_method(shuriken::dex::Method &);

void print_field(shuriken::dex::Field &);

void print_code(shuriken::dex::Method &);

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
            std::unique_ptr<shuriken::dex::Dex> & dex = dex_file.value();
            print_classes(*dex);
        }
    }
}

void parse_dex(std::string &dex_file_str) {
    dex_file = shuriken::dex::Dex::create_from_file(dex_file_str);

    if (!dex_file.has_value()) {
        const auto& error = dex_file.error();

        fmt::println("ERROR generating the DEX object {}", error.get_message());

        std::exit(3);
    }
}

void print_classes(shuriken::dex::Dex &dex) {
    for (auto & cls : dex.get_classes()) {
        fmt::println("CLASS: {}", cls.get_name());
        fmt::println("CANONICAL NAME: {}", cls.get_canonical_name());
        fmt::println("DALVIK NAME: {}", cls.get_dalvik_name());

        if (methods) {
            for (auto & method : cls.get_methods())
                print_method(method);
        }
        if (fields) {
            for (auto & field : cls.get_fields())
                print_field(field);
        }
    }
}

void print_method(shuriken::dex::Method &method) {
    fmt::println("\tMETHOD: {}", method.get_descriptor());
}

void print_field(shuriken::dex::Field &field) {
    fmt::println("\tFIELD: {}", field.get_descriptor());
}