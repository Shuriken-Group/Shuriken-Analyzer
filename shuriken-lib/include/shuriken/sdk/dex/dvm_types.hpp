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

class DVMFundamental {
private:
    std::reference_wrapper<DVMFundamentalProvider> dvm_fundamental_provider;
public:
    DVMFundamental(DVMFundamentalProvider &);
    ~DVMFundamental() = default;

    DVMFundamental(const DVMFundamental&) = delete;
    DVMFundamental& operator=(const DVMFundamental&) = delete;


    types::type_e get_type() const;

    std::string_view get_dalvik_format() const;

    std::string get_dalvik_format_string() const;

    std::string_view get_canonical_name() const;

    std::string get_canonical_name_string() const;

    types::fundamental_e get_fundamental_type() const;
};

class DVMClass {
private:
    std::reference_wrapper<DVMClassProvider> dvm_class_provider;
public:
    DVMClass(DVMClassProvider&);
    ~DVMClass() = default;

    DVMClass(const DVMClass&) = delete;
    DVMClass& operator=(const DVMClass&) = delete;

    types::type_e get_type() const;

    std::string_view get_dalvik_format() const;

    std::string get_dalvik_format_string() const;

    std::string_view get_canonical_name() const;

    std::string get_canonical_name_string() const;
};

class DVMArray {
private:
    std::reference_wrapper<DVMArrayProvider> dvm_array_provider;
public:
    DVMArray(DVMArrayProvider&);
    ~DVMArray() = default;

    DVMArray(const DVMArray&) = delete;
    DVMArray& operator=(const DVMArray&) = delete;

    types::type_e get_type() const;

    std::string_view get_dalvik_format() const;

    std::string get_dalvik_format_string() const;

    std::string_view get_canonical_name() const;

    std::string get_canonical_name_string() const;

    size_t get_array_depth() const;

    const DVMType* get_base_type() const;
};

/**
 * @param type type to obtain its main type_e
 * @return type_e of provided object
 */
types::type_e get_type(const DVMType& type);

/**
 * @param type type to obtain its name in dalvik format
 * @return dalvik format of type as string_view
 */
std::string_view get_dalvik_format_string(const DVMType& type);

/**
 * @param type type to obtain its name in dalvik format
 * @return dalvik format of type as string
 */
std::string get_dalvik_format(const DVMType& type);

/**
 * @param type type to obtain its name in canonical format
 * @return canonical format of type as string
 */
std::string_view get_canonical_name(const DVMType& type);

/**
 * @param type type to obtain its name in canonical format
 * @return canonical format of type as string
 */
std::string get_canonical_name_string(const DVMType& type);

const DVMFundamental * as_fundamental(const DVMType& type);

const DVMClass * as_class(const DVMType& type);

const DVMArray * as_array(const DVMType& type);

} // namespace dex
} // namespace shuriken