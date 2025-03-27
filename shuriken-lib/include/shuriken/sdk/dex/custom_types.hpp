//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <span>
#include <variant>
#include <unordered_map>
#include <set>

#include <shuriken/sdk/common/deref_iterator_range.hpp>
#include <shuriken/sdk/dex/constants.hpp>

namespace shuriken {
    namespace dex {

        class Class;
        class ExternalClass;
        class Method;
        class ExternalMethod;
        class Field;
        class ExternalField;
        class Instruction;
        class DVMTypes;

        using class_t = std::reference_wrapper<Class>;
        using method_t = std::reference_wrapper<Method>;
        using field_t = std::reference_wrapper<Field>;
        using instruction_t = std::reference_wrapper<Instruction>;
        using dvmtypes_t = std::reference_wrapper<DVMTypes>;

        using externalclass_t = std::reference_wrapper<ExternalClass>;
        using externalmethod_t = std::reference_wrapper<ExternalMethod>;
        using externalfield_t = std::reference_wrapper<ExternalField>;



        using class_external_class_t = std::variant<class_t, externalclass_t, std::monostate>;

        // A type which points to an instruction in a given index, from
        // a specific method, from a specific class
        using class_method_idx_t = std::tuple<class_t, method_t, std::uint64_t>;
        using span_class_method_idx_t = std::span<class_method_idx_t>;
        using span_class_method_idx_iterator_t = span_class_method_idx_t::iterator;

        using class_field_idx_t = std::tuple<class_t, field_t, std::uint64_t>;
        using span_class_field_idx_t = std::span<class_field_idx_t>;
        using span_class_field_idx_iterator_t = span_class_field_idx_t::iterator;

        using class_idx_t = std::tuple<class_t, std::uint64_t>;
        using span_class_idx_t = std::span<class_idx_t>;
        using span_class_idx_iterator_t = span_class_idx_t::iterator;

        using span_class_external_class_t = std::span<class_external_class_t>;
        using span_class_external_class_iterator_t = span_class_external_class_t::iterator;

        /**
         * Maps a class to a set of reference points (ref_type, method, instruction offset)
         * Used to represent cross-references between classes
         * - Key: The referenced/referencing class
         * - Value: Set of tuples containing:
         *    - The type of reference (instantiation, method call, etc.)
         *    - The method containing the reference
         *    - The offset of the instruction within the method
         */
        using classxref_t = std::unordered_map<class_t,
                std::set<std::tuple<types::ref_type,
                        method_t,
                        std::uint64_t>>>;
        using classxref_iterator_t = classxref_t::iterator;

        using method_idx_t = std::pair<method_t, std::uint64_t>;
        using span_method_idx_t = std::span<method_idx_t>;
        using span_method_idx_iterator_t = span_method_idx_t::iterator;

        using methods_ref_t = std::span<method_t>;
        using method_deref_iterator_t = deref_iterator_range<methods_ref_t>;


        using fields_ref_t = std::span<field_t>;
        using fields_deref_iterator_t = deref_iterator_range<fields_ref_t>;

        using instruction_list_t = std::span<instruction_t>;
        using instruction_list_deref_iterator_t = deref_iterator_range<instruction_list_t>;

        using dvmtypes_list_t = std::span<dvmtypes_t>;
        using dvmtypes_list_deref_iterator_t = deref_iterator_range<dvmtypes_list_t>;

    } // namespace dex
} // namespace shuriken