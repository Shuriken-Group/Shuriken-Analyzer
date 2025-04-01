//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <variant>
#include <span>

namespace shuriken {
namespace dex {

class DVMFundamentalProvider;
class DVMClassProvider;
class DVMArrayProvider;

using DVMTypeProvider = std::variant<DVMFundamentalProvider, DVMClassProvider, DVMArrayProvider>;
using dvmtypeprovider_t = std::reference_wrapper<DVMTypeProvider>;
using dvmtypesprovider_list_t = std::span<dvmtypeprovider_t>;
using dvmtypesprovider_list_deref_iterator_t = deref_iterator_range<dvmtypesprovider_list_t>;

} // namespace dex
} // namespace shuriken