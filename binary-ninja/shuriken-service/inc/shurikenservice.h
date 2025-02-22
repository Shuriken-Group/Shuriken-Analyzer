#pragma once

#include "binaryninjaapi.h"
#include <shuriken/shuriken_cpp_core.h>

#if defined(_WIN32) || defined(__WIN32__)
#ifdef SHURIKENSERVICE_EXPORTS
#define SHURIKENSERVICE_API __declspec(dllexport)
#else
#define SHURIKENSERVICE_API __declspec(dllimport)
#endif
#else
#define SHURIKENSERVICE_API
#endif

class IShurikenView {
public:
    IShurikenView() = default;
    virtual ~IShurikenView() = default;

    virtual const shurikenapi::IDex &getParser() const = 0;
    virtual uint32_t getStringOffset(int64_t id) const = 0;
};

class IShurikenService {
public:
    IShurikenService() = default;
    virtual ~IShurikenService() = default;

    virtual bool registerView(BinaryNinja::BinaryView *view) = 0;
    virtual const IShurikenView &getView(BinaryNinja::BinaryView *view) = 0;
};

SHURIKENSERVICE_API IShurikenService &GetShurikenService();