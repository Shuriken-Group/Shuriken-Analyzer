//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

namespace shuriken::dex {

class DexEngine;
class Class;
class Method;
class Field;

class XrefGenerator {
private:
    DexEngine * current_dex_engine;

    void analyze_class(Class * current_class);
    void analyze_method(Method * method);
public:
    XrefGenerator() = default;
    ~XrefGenerator() = default;

    void analyze_xrefs(DexEngine * dex_engine);
};
}