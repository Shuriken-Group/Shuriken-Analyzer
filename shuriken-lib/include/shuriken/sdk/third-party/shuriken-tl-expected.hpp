//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#if __cplusplus >= 202300L
// C++23 or later - use the standard library version
    #include <expected>
    namespace shuriken {
    namespace third_party {
        template<typename T, typename E>
        using expected = std::expected<T, E>;

        template<typename E>
        using unexpected = std::unexpected<E>;

        template<typename E>
        auto make_unexpected(E&& e) {
            return std::unexpected<std::decay_t<E>>(std::forward<E>(e));
        }
    }
    }
#else
// Pre-C++23 - use the third-party implementation
#include "./tl/expected.hpp"
namespace shuriken {
    namespace third_party {
        template<typename T, typename E>
        using expected = tl::expected<T, E>;

        template<typename E>
        using unexpected = tl::unexpected<E>;

        template<typename E>
        auto make_unexpected(E&& e) {
            return tl::make_unexpected(std::forward<E>(e));
        }
    }
}
#endif