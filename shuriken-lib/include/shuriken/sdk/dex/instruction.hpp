//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>

namespace shuriken {
namespace dex {
class DexInstructionProvider;

class InstructionProvider {
private:
    std::reference_wrapper<DexInstructionProvider> dex_instruction_provider;
public:
    // constructors & destructors
    InstructionProvider(DexInstructionProvider&);
    ~InstructionProvider() = default;

    InstructionProvider(const InstructionProvider&) = delete;
    InstructionProvider& operator=(const InstructionProvider&) = delete;
};
}
}