#pragma once

#include "binaryninjaapi.h"
#include <vector>

using BinaryNinja::BinaryReader;
using BinaryNinja::BinaryView;
using BinaryNinja::DataRenderer;
using BinaryNinja::DisassemblyTextLine;
using BinaryNinja::Function;
using BinaryNinja::InstructionTextToken;
using BinaryNinja::Type;

class StringTable : public DataRenderer {

    bool IsValidForData(BinaryView *data, uint64_t addr, Type *type, std::vector<std::pair<Type *, size_t>> &context) override;
    std::vector<DisassemblyTextLine> GetLinesForData(BinaryView *data, uint64_t addr, Type *type,
                                                     const std::vector<InstructionTextToken> &prefix, size_t width,
                                                     std::vector<std::pair<Type *, size_t>> &context, const std::string &language = std::string()) override;
};
