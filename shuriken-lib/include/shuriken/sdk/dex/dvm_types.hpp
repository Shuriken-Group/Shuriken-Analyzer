//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>

namespace shuriken {
namespace dex {
class DVMTypeProvider;
class DVMFundamentalProvider;
class DVMClassProvider;
class DVMArrayProvider;

class DVMType {
private:
    std::reference_wrapper<DVMTypeProvider> dvm_type_provider;
public:
    DVMType(DVMTypeProvider &);
    virtual ~DVMType() = default;

    DVMType(const DVMType&) = delete;
    DVMType& operator=(const DVMType&) = delete;

    /**
     * @return current type, used to differentiate between objects.
     */
    virtual types::type_e get_type() const = 0;

    virtual std::string_view get_dalvik_format() const = 0;

    virtual std::string get_dalvik_format_string() const = 0;

    virtual std::string_view get_canonical_name() const = 0;

    virtual std::string get_canonical_name_string() const = 0;
};

class DVMFundamental : public DVMType {
private:
    std::reference_wrapper<DVMFundamentalProvider> dvm_fundamental_provider;
public:
    DVMFundamental(DVMFundamentalProvider &);
    ~DVMFundamental() = default;

    DVMFundamental(const DVMFundamental&) = delete;
    DVMFundamental& operator=(const DVMFundamental&) = delete;


    types::type_e get_type() const override;

    std::string_view get_dalvik_format() const override;

    std::string get_dalvik_format_string() const override;

    std::string_view get_canonical_name() const override;

    std::string get_canonical_name_string() const override;

    types::fundamental_e get_fundamental_type() const;
};

class DVMClass : public DVMType {
private:
    std::reference_wrapper<DVMClassProvider> dvm_class_provider;
public:
    DVMClass(DVMClassProvider&);
    ~DVMClass() = default;

    DVMClass(const DVMClass&) = delete;
    DVMClass& operator=(const DVMClass&) = delete;

    types::type_e get_type() const override;

    std::string_view get_dalvik_format() const override;

    std::string get_dalvik_format_string() const override;

    std::string_view get_canonical_name() const override;

    std::string get_canonical_name_string() const override;
};

class DVMArray : public DVMType {
private:
    std::reference_wrapper<DVMArrayProvider> dvm_array_provider;
public:
    DVMArray(DVMArrayProvider&);
    ~DVMArray() = default;

    DVMArray(const DVMArray&) = delete;
    DVMArray& operator=(const DVMArray&) = delete;

    types::type_e get_type() const override;

    std::string_view get_dalvik_format() const override;

    std::string get_dalvik_format_string() const override;

    std::string_view get_canonical_name() const override;

    std::string get_canonical_name_string() const override;

    size_t get_array_depth() const;

    const DVMType* get_base_type() const;
};

} // namespace dex
} // namespace shuriken