#include <shuriken/internal/io/shurikenstream.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace shuriken {
namespace io {
namespace detail {

    // Implementation interface
    class StreamImpl {
    public:
        virtual ~StreamImpl() = default;
        virtual std::streampos position() const = 0;
        virtual void seek(std::streampos pos) = 0;
        virtual void seek(std::streamoff offset, std::ios::seekdir dir) = 0;
        virtual bool good() const = 0;
        virtual bool eof() const = 0;
        virtual void read_bytes(char* buffer, size_t length) = 0;
    };

    // Implementation for file streams
    class FileStreamImpl : public StreamImpl {
    private:
        std::ifstream stream;

    public:
        explicit FileStreamImpl(std::string_view path)
                : stream(path.data(), std::ios::binary) {}

        std::streampos position() const override {
            return const_cast<std::ifstream&>(stream).tellg();
        }

        void seek(std::streampos pos) override {
            stream.seekg(pos);
        }

        void seek(std::streamoff offset, std::ios::seekdir dir) override {
            stream.seekg(offset, dir);
        }

        bool good() const override {
            return stream.good();
        }

        bool eof() const override {
            return stream.eof();
        }

        void read_bytes(char* buffer, size_t length) override {
            stream.read(buffer, length);
        }
    };

    // Implementation for memory streams
    class MemoryStreamImpl : public StreamImpl {
    private:
        std::istringstream stream;

    public:
        explicit MemoryStreamImpl(const std::string& data)
                : stream(data) {}

        std::streampos position() const override {
            return const_cast<std::istringstream&>(stream).tellg();
        }

        void seek(std::streampos pos) override {
            stream.seekg(pos);
        }

        void seek(std::streamoff offset, std::ios::seekdir dir) override {
            stream.seekg(offset, dir);
        }

        bool good() const override {
            return stream.good();
        }

        bool eof() const override {
            return stream.eof();
        }

        void read_bytes(char* buffer, size_t length) override {
            stream.read(buffer, length);
        }
    };

} // namespace detail

// Implementation of ShurikenStream methods
ShurikenStream::ShurikenStream() : impl(nullptr) {}

ShurikenStream::~ShurikenStream() = default;

ShurikenStream::ShurikenStream(ShurikenStream&& other) noexcept
        : impl(std::move(other.impl)) {
}

ShurikenStream& ShurikenStream::operator=(ShurikenStream&&) noexcept = default;

error::Result<ShurikenStream> ShurikenStream::from_file(std::string_view path) {
    auto impl = std::make_unique<detail::FileStreamImpl>(path);
    if (!impl->good()) {
        return error::make_error<ShurikenStream>(
                error::ErrorCode::FileNotFound,
                "Failed to open file: " + std::string(path)
        );
    }

    ShurikenStream stream;
    stream.impl = std::move(impl);
    return error::make_success(std::move(stream));
}

error::Result<ShurikenStream> ShurikenStream::from_memory(const std::string& data) {
    ShurikenStream stream;
    stream.impl = std::make_unique<detail::MemoryStreamImpl>(data);
    return error::make_success(std::move(stream));
}

void ShurikenStream::read_bytes(char* buffer, size_t length) {
    impl->read_bytes(buffer, length);
}

std::string ShurikenStream::read_string(size_t length) {
    std::string result(length, '\0');
    impl->read_bytes(&result[0], length);
    return result;
}

std::string ShurikenStream::read_string_at(std::streampos position, size_t length) {
    impl->seek(position);
    return read_string(length);
}

std::streampos ShurikenStream::position() const {
    return impl->position();
}

void ShurikenStream::seek(std::streampos pos) {
    impl->seek(pos);
}

void ShurikenStream::seek(std::streamoff offset, std::ios_base::seekdir dir) {
    impl->seek(offset, dir);
}

bool ShurikenStream::good() const {
    return impl && impl->good();
}

bool ShurikenStream::eof() const {
    return impl && impl->eof();
}

std::uint64_t ShurikenStream::read_uleb128() {
    std::uint64_t value = 0;
    unsigned shift = 0;
    std::int8_t byte_read;

    do {
        byte_read = read<std::int8_t>();
        value |= static_cast<std::uint64_t>(byte_read & 0x7F) << shift;
        shift += 7;
    } while (byte_read & 0x80);

    return value;
}

std::int64_t ShurikenStream::read_sleb128() {
    std::int64_t value = 0;
    unsigned shift = 0;
    std::int8_t byte_read;

    do {
        byte_read = read<std::int8_t>();
        value |= static_cast<std::uint64_t>(byte_read & 0x7F) << shift;
        shift += 7;
    } while (byte_read & 0x80);

    // sign extend negative numbers
    if ((byte_read & 0x40))
        value |= static_cast<std::int64_t>(-1) << shift;

    return value;
}

std::string ShurikenStream::read_dex_string(std::int64_t offset) {
    auto current_offset = position();

    seek(static_cast<std::streampos>(offset));

    [[maybe_unused]] auto utf16_size = read_uleb128();

    std::string new_str;
    char c;
    while ((c = read<char>()) && c != 0) {
        new_str.push_back(c);
    }

    seek(current_offset);

    return new_str;
}

} // namespace io
} // namespace shuriken