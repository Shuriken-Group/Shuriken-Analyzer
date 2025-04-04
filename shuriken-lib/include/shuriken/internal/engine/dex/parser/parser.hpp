//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <shuriken/internal/engine/dex/parser/dex_header.hpp>


#include <shuriken/internal/io/shurikenstream.hpp>
#include <shuriken/sdk/common/error.hpp>

namespace shuriken {
namespace dex {

class Parser {
private:
    DexHeader header_;

public:
    Parser() = default;
    ~Parser() = default;

    error::VoidResult parse(io::ShurikenStream& stream);
};
}
}