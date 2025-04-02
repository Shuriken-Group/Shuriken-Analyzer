//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <memory>

#include <shuriken/internal/engine/dex/dex_engine.hpp>

#include "shuriken/sdk/dex/class.hpp"
#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/sdk/dex/field.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/internal/providers/dex/dex_class_provider.hpp"
#include "shuriken/internal/providers/dex/dex_method_provider.hpp"
#include "shuriken/internal/providers/dex/dex_field_provider.hpp"
#include "shuriken/internal/providers/dex/dvm_types_provider.hpp"
#include "shuriken/internal/providers/dex/dvm_prototypes_provider.hpp"
#include "shuriken/internal/providers/dex/custom_types.hpp"


using namespace shuriken::dex;

class DexEngine::Impl {
public:
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

    // cache of all the previous vectors
    std::vector<std::reference_wrapper<Class>> ref_sdk_classes;
    std::vector<std::reference_wrapper<DexClassProvider>> ref_dex_class_providers;
    std::vector<std::reference_wrapper<Method>> ref_sdk_methods;
    std::vector<std::reference_wrapper<DexMethodProvider>> ref_dex_methods_providers;
    std::vector<std::reference_wrapper<Field>> ref_sdk_fields;
    std::vector<std::reference_wrapper<DexFieldProvider>> ref_dex_fields_providers;
    std::vector<std::reference_wrapper<DVMPrototype>> ref_sdk_prototypes;
    std::vector<std::reference_wrapper<DVMPrototypeProvider>> ref_dex_prototypes_providers;
    std::vector<std::reference_wrapper<DVMType>> ref_sdk_dvmtypes;
    std::vector<std::reference_wrapper<DVMTypeProvider>> ref_dex_type_providers;

    Impl() = default;
    ~Impl() = default;
};

DexEngine::DexEngine(shuriken::io::ShurikenStream stream) : shuriken_stream(std::move(stream)),
            pimpl(std::make_unique<DexEngine::Impl>()) {
}



