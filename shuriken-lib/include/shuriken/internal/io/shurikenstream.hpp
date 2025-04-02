//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <cstdint>

#include <shuriken/sdk/common/error.hpp>

namespace shuriken {
    namespace io {

// Forward declarations for stream types
        namespace detail {
            class StreamImpl;
        }

        class ShurikenStream {
        private:
            std::unique_ptr<detail::StreamImpl> impl;

        public:
            // Constructors and destructor
            ShurikenStream();
            ~ShurikenStream();

            // Disable copy, enable move
            ShurikenStream(const ShurikenStream&) = delete;
            ShurikenStream& operator=(const ShurikenStream&) = delete;
            ShurikenStream(ShurikenStream&&) noexcept;
            ShurikenStream& operator=(ShurikenStream&&) noexcept;

            // Factory methods
            static error::Result<ShurikenStream> from_file(const std::string& path);
            static error::Result<ShurikenStream> from_memory(const std::string& data);

            // Stream operations
            template <typename T>
            T read() {
                T value;
                read_bytes(reinterpret_cast<char*>(&value), sizeof(T));
                return value;
            }

            template <typename T>
            T read_at(std::streampos position) {
                seek(position);
                return read<T>();
            }

            void read_bytes(char* buffer, size_t length);

            std::string read_string(size_t length);
            std::string read_string_at(std::streampos position, size_t length);

            // Position methods
            std::streampos position() const;
            void seek(std::streampos pos);
            void seek(std::streamoff offset, std::ios_base::seekdir dir);

            // Status checks
            bool good() const;
            bool eof() const;

            // Special methods for common data formats
            std::uint64_t read_uleb128();
            std::int64_t read_sleb128();
            std::string read_dex_string(std::int64_t offset);
        };

    } // namespace io
} // namespace shuriken