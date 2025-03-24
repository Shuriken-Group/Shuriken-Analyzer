//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <span>

namespace shuriken {
namespace dex {

class Class;
class Method;
class Field;

class Instruction;

// A type which points to an instruction in a given index, from
// a specific method, from a specific class
using class_method_idx_t = std::tuple<Class*, Method*, std::uint64_t>;
using span_class_method_idx_t = std::span<class_method_idx_t>;
using span_class_method_idx_iterator_t = span_class_method_idx_t::iterator;

using class_field_idx_t = std::tuple<Class*, Field*, std::uint64_t>;
using span_class_field_idx_t = std::span<class_field_idx_t>;
using span_class_field_idx_iterator_t = span_class_field_idx_t::iterator;

using class_idx_t = std::tuple<Class*, std::uint64_t>;
using span_class_idx_t = std::span<class_idx_t>;
using span_class_idx_iterator_t = span_class_idx_t::iterator;




using instruction_list_t = std::span<Instruction*>;
using instruction_list_iterator_t = instruction_list_t::iterator;


} // namespace dex
} // namespace shuriken