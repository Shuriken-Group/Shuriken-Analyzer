//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <utility>

namespace shuriken {
/// A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template<typename T>
class iterator_range {
    T begin_iterator, end_iterator;
public:
    //TODO: Add SFINAE to test that the Container's iterators match the range's
    //      iterators.
    template<typename Container>
    iterator_range(Container &&c)
    //TODO: Consider ADL/non-member begin/end calls.
            : begin_iterator(c.begin()), end_iterator(c.end()) {}
    iterator_range(T begin_iterator, T end_iterator)
            : begin_iterator(std::move(begin_iterator)),
              end_iterator(std::move(end_iterator)) {}

    T begin() const { return begin_iterator; }
    T end() const { return end_iterator; }
    bool empty() const { return begin_iterator == end_iterator; }
};

/// Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template<class T>
iterator_range<T> make_range(T x, T y) {
    return iterator_range<T>(std::move(x), std::move(y));
}

template<typename T>
iterator_range<T> make_range(std::pair<T, T> p) {
    return iterator_range<T>(std::move(p.first), std::move(p.second));
}

} // namespace shuriken