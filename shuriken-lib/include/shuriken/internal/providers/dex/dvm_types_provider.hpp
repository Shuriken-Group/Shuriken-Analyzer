//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

namespace shuriken {
namespace dex {

class DVMTypeProvider {};

class DVMFundamentalProvider : public DVMTypeProvider {};

class DVMClassProvider : public DVMTypeProvider {};

class DVMArrayProvider : public DVMTypeProvider {};

} // namespace dex
} // namespace shuriken