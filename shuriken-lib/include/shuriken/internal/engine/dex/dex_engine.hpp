//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <vector>
#include <memory>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/internal/providers/dex/custom_types.hpp>
#include <shuriken/internal/io/shurikenstream.hpp>

namespace shuriken {
namespace dex {

class Class;
class DexClassProvider;
class Method;
class DexMethodProvider;
class Field;
class DexFieldProvider;

class DVMPrototype;
class DVMPrototypeProvider;

class DexEngine {
private:
    class Impl; // Forward declaration of implementation class
    std::unique_ptr<Impl> pimpl; // The pointer to implementation
    io::ShurikenStream shuriken_stream;
public:
    // Constructor/destructor
    DexEngine(io::ShurikenStream stream);
    ~DexEngine() = default;

    // Move operations
    DexEngine(DexEngine&&) noexcept = delete;
    DexEngine& operator=(DexEngine&&) noexcept = delete;

    // Disable copy
    DexEngine(const DexEngine&) = delete;
    DexEngine& operator=(const DexEngine&) = delete;
};
}
}