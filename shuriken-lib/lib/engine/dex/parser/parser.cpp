
#include <shuriken/internal/engine/dex/parser/parser.hpp>

using namespace shuriken::dex;

shuriken::error::VoidResult Parser::parse(io::ShurikenStream& stream) {
    if (!header_.parse(stream)) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex header");
    }

    return error::make_success();
}
