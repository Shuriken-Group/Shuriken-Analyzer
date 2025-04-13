//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <variant>

#include <shuriken/sdk/third-party/shuriken-tl-expected.hpp>

namespace shuriken {
namespace error {

// Define error types/codes
enum class ErrorCode {
    Success = 0,
    FileNotFound,
    InvalidFile,
    ParseError,
    MemoryError,
    UnknownError,
    // Add more specific error codes as needed
    ParsingError,
    InvalidType,
};

// Error class to hold error information
class Error {
private:
    ErrorCode code;
    std::string message;

public:
    Error() : code(ErrorCode::Success), message("") {}

    Error(ErrorCode code, std::string_view message = "")
            : code(code), message(std::string(message)) {}

    ErrorCode get_code() const { return code; }

    const std::string& get_message() const { return message; }

    bool is_success() const { return code == ErrorCode::Success; }

    // Allow implicit conversion to bool for easy checking
    operator bool() const { return !is_success(); }
};

// Type alias for expected results
template <typename T>
using Result =third_party::expected<T, Error>;

template <typename T>
Result<T> make_success(T value) {
    return Result<T>(std::move(value));
}

// Helper for creating error results
template <typename T>
Result<T> make_error(ErrorCode code, std::string_view message = "") {
    return third_party::make_unexpected(Error(code, message));
}

// For void returns
using VoidResult = Result<std::monostate>;

inline VoidResult make_success() {
    return VoidResult(std::monostate{});
}

inline VoidResult make_error(ErrorCode code, std::string_view message = "") {
    return third_party::make_unexpected(Error(code, message));
}

} // namespace error
} // namespace shuriken