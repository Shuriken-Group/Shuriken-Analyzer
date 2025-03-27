//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>

namespace shuriken {
namespace dex {
class DexInstructionProvider;

class Instruction {
private:
    std::reference_wrapper<DexInstructionProvider> dex_instruction_provider;
public:
    // constructors & destructors
    Instruction(DexInstructionProvider&);
    ~Instruction() = default;

    Instruction(const Instruction&) = delete;
    Instruction& operator=(const Instruction&) = delete;
};
}
}