//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <vector>
#include <memory>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/internal/providers/dex/custom_types.hpp>

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
    // ownership of classes
    std::vector<std::unique_ptr<Class>> sdk_classes;
    std::vector<std::unique_ptr<DexClassProvider>> dex_class_providers;

    // ownership of methods
    std::vector<std::unique_ptr<Method>> sdk_methods;
    std::vector<std::unique_ptr<DexMethodProvider>> dex_methods_providers;

    // ownership of fields
    std::vector<std::unique_ptr<Field>> sdk_fields;
    std::vector<std::unique_ptr<DexFieldProvider>> dex_fields_providers;

    // ownership of prototypes
    std::vector<std::unique_ptr<DVMPrototype>> sdk_prototypes;
    std::vector<std::unique_ptr<DVMPrototypeProvider>> dex_prototypes_providers;

    // ownership of types
    std::vector<std::unique_ptr<DVMType>> sdk_dvmtypes;
    std::vector<std::unique_ptr<DVMTypeProvider>> dex_type_providers;

    


};
}
}