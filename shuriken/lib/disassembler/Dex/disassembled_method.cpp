//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file disassembled_method.cpp

#include "shuriken/disassembler/Dex/disassembled_method.h"
#include <algorithm>// for std::find
#include <cctype>
#include <iomanip>
#include <iterator>// for std::distance
#include <sstream>

using namespace shuriken::disassembler::dex;

DisassembledMethod::DisassembledMethod(shuriken::parser::dex::MethodID *method_id,
                                       std::uint16_t n_of_registers,
                                       exceptions_data_t &exceptions,
                                       instructions_t &instructions,
                                       shuriken::dex::TYPES::access_flags access_flags) : method_id(method_id),
                                                                                          n_of_registers(n_of_registers),
                                                                                          exception_information(std::move(exceptions)),
                                                                                          instructions(std::move(instructions)),
                                                                                          access_flags(access_flags) {
    instructions_raw.reserve(this->instructions.size());
    for (const auto &instr: this->instructions) {
        instructions_raw.emplace_back(instr.get());
    }
}

shuriken::parser::dex::MethodID *DisassembledMethod::get_method_id() {
    return method_id;
}

std::uint16_t DisassembledMethod::get_number_of_registers() const {
    return n_of_registers;
}

size_t DisassembledMethod::get_number_of_exceptions() const {
    return exception_information.size();
}

size_t DisassembledMethod::get_number_of_instructions() const {
    return instructions.size();
}

it_exceptions_data DisassembledMethod::get_exceptions() {
    return make_range(exception_information.begin(), exception_information.end());
}

it_instructions DisassembledMethod::get_instructions() {
    return make_range(instructions_raw.begin(), instructions_raw.end());
}

const instructions_raw_t &DisassembledMethod::get_instructions_container() {
    return instructions_raw;
}

std::span<Instruction *> DisassembledMethod::get_ref_to_instructions(size_t init, size_t end) {

    if (end > instructions_raw.back()->get_address())
        throw std::runtime_error{"Error, last index out of bounds"};
    if (init > end)
        throw std::runtime_error{"Error, init must be lower to end"};

    auto first = instructions_raw.end();
    auto second = instructions_raw.end();

    for (auto it = instructions_raw.begin(); it != instructions_raw.end(); ++it) {
        if ((*it)->get_address() == init) {
            first = it;
        }
        if ((*it)->get_address() == end) {
            second = it;
        }
        if (first != instructions_raw.end() && second != instructions_raw.end()) {
            break;
        }
    }

    if (first == instructions_raw.end() || second == instructions_raw.end())
        throw std::runtime_error{"Error, some index is out of bounds"};

    std::span<Instruction *> block{&(*first), static_cast<size_t>(std::distance(first, second) + 1)};
    return block;
}

std::string_view DisassembledMethod::print_method(bool print_address) {
    if (method_string.empty()) {
        std::stringstream output;
        std::string access_flags_str = shuriken::dex::Utils::get_types_as_string(access_flags);
        std::transform(access_flags_str.begin(),
                       access_flags_str.end(),
                       access_flags_str.begin(),
                       [](unsigned char c) {
                           if (c == '|')
                               return ' ';
                           return static_cast<char>(tolower(c));
                       });
        output << ".method " << access_flags_str << " ";
        output << method_id->dalvik_name_format() << '\n';
        output << ".registers " << std::to_string(n_of_registers) << '\n';
        std::uint64_t id = 0;
        for (auto *instr: instructions_raw) {
            /// check the information for showing exception
            for (const auto &exception: exception_information) {
                /// avoid printing future try-catch handlers
                if (id < exception.try_value_start_addr)
                    continue;

                if (id == exception.try_value_start_addr)
                    output << ".try_start_" << (exception.try_value_start_addr / 2) << "\n";
                if (id == exception.try_value_end_addr) {
                    output << ".try_end_" << (exception.try_value_end_addr / 2) << "\n";
                    for (const auto &catch_data: exception.handler) {
                        output << ".catch " << catch_data.handler_type->get_raw_type();
                        output << " {.try_start_" << (exception.try_value_start_addr / 2) << " .. ";
                        output << ".try_end_" << (exception.try_value_end_addr / 2) << "}";
                        output << " :catch_" << (catch_data.handler_start_addr / 2) << "\n";
                    }
                }

                for (const auto &catch_data: exception.handler) {
                    if (catch_data.handler_start_addr == id) {
                        output << ":catch_" << (catch_data.handler_start_addr / 2) << '\n';
                    }
                }
            }
            if (print_address)
                output << std::hex << std::setw(8) << std::setfill('0') << id;
            output << ' ';
            /// now print the instruction
            instr->print_instruction(output);
            id += instr->get_instruction_length();
            output << '\n';
        }
        output << ".end method";
        method_string = output.str();
    }
    return method_string;
}
