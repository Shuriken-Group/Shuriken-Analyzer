
#include <shuriken/internal/engine/dex/parser/parser.hpp>
#include <shuriken/sdk/dex/constants.hpp>

using namespace shuriken::dex;

shuriken::error::VoidResult Parser::parse(io::ShurikenStream &stream) {
    if (!header_.parse(stream)) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex header");
    }

    if (!parse_string_pool(stream, header_.get_string_ids_off(), header_.get_string_ids_size())) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex strings");
    }

    if (!parse_types(stream, header_.get_type_ids_off(), header_.get_type_ids_size())) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex types");
    }

    if (!parse_protos(stream, header_.get_proto_ids_off(), header_.get_proto_ids_size())) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex protos");
    }

    if (!parse_fields(stream, header_.get_field_ids_off(), header_.get_field_ids_size())) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex fields");
    }

    if (!parse_methods(stream, header_.get_method_ids_off(), header_.get_method_ids_size())) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex methods");
    }

    if (!parse_classes(stream, header_.get_class_defs_off(), header_.get_class_defs_size())) {
        return error::make_error(error::ErrorCode::ParseError, "Error reading the dex classes");
    }

    return error::make_success();
}

bool
Parser::parse_string_pool(shuriken::io::ShurikenStream &stream, std::uint32_t strings_offset,
                          std::uint32_t n_of_strings) {
    std::uint32_t str_offset;// we will read offsets

    auto current_offset = stream.position();
    stream.seek(static_cast<std::streamoff>(strings_offset));

    for (std::uint32_t I = 0; I < n_of_strings; I++) {
        str_offset = stream.read<std::uint32_t>();

        string_pool.emplace_back(stream.read_dex_string(str_offset));
    }

    stream.seek(current_offset);

    return stream.good();
}

DVMTypeProvider *Parser::parse_type(std::string_view name) {
    switch (name.at(0)) {
        case 'Z':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::BOOLEAN));
        case 'B':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::BYTE));
        case 'C':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::CHAR));
        case 'D':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::DOUBLE));
        case 'F':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::FLOAT));
        case 'I':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::INT));
        case 'J':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::LONG));
        case 'S':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::SHORT));
        case 'V':
            return new DVMTypeProvider(DVMFundamentalProvider(name, types::fundamental_e::VOID));
        case 'L':
            return new DVMTypeProvider(DVMClassProvider(name));
        case '[': {
            size_t depth = 0;
            for (const auto &c: name) {
                if (c == '[') depth++;
                else
                    break;
            }
            std::string_view aux(name.begin() + depth, name.end());
            DVMTypeProvider *aux_type = parse_type(aux);
            return new DVMTypeProvider(std::in_place_type<DVMArrayProvider>,
                                       name, depth, aux_type);
        }
        default:
            return nullptr;
    }
}

bool Parser::parse_types(shuriken::io::ShurikenStream &stream, std::uint32_t types_offset, std::uint32_t n_of_types) {
    auto current_offset = stream.position();
    std::uint32_t type_id;

    stream.seek(types_offset);

    for (size_t I = 0; I < n_of_types; I++) {
        type_id = stream.read<std::uint32_t>();

        if (type_id > string_pool.size()) return false;

        std::unique_ptr<DVMTypeProvider> type(parse_type(string_pool[type_id]));
        std::unique_ptr<DVMType> base_type;

        DVMTypeProvider &type_provider = *(type);

        if (::get_type(type_provider) == types::type_e::FUNDAMENTAL) {
            DVMFundamentalProvider *fundamental = ::as_fundamental(type_provider);
            base_type = std::make_unique<DVMType>(DVMFundamental(*fundamental));
        } else if (::get_type(type_provider) == types::type_e::CLASS) {
            DVMClassProvider *class_provider = ::as_class(type_provider);
            base_type = std::make_unique<DVMType>(DVMClass(*class_provider));
        } else {
            DVMArrayProvider *array_provider = ::as_array(type_provider);
            base_type = std::make_unique<DVMType>(DVMArray(*array_provider));
        }

        types_pool.emplace_back(std::move(type));
        dvm_types_pool.emplace_back(std::move(base_type));
    }

    stream.seek(current_offset);
    return stream.good();
}

std::vector<dvmtype_t> Parser::parse_parameters(io::ShurikenStream &stream,
                                                std::uint32_t parameters_off) {
    auto current_offset = stream.position();
    std::vector<dvmtype_t> parameters;
    std::uint32_t n_parameters;
    std::uint16_t type_id;

    if (!parameters_off) return parameters;

    stream.seek(parameters_off);

    n_parameters = stream.read<std::uint32_t>();

    for (std::uint32_t I = 0; I < n_parameters; I++) {
        type_id = stream.read<std::uint16_t>();
        DVMType &type = *(dvm_types_pool[type_id].get());
        parameters.emplace_back(type);
    }

    stream.seek(current_offset);

    return parameters;
}

bool
Parser::parse_protos(shuriken::io::ShurikenStream &stream, std::uint32_t protos_offset, std::uint32_t n_of_protos) {
    auto current_offset = stream.position();
    std::uint32_t shorty_idx = 0,//! id for prototype string
    return_type_idx = 0, //! id for type of return
    parameters_off = 0;  //! offset of parameters

    stream.seek(protos_offset);

    for (size_t I = 0; I < n_of_protos; ++I) {
        shorty_idx = stream.read<std::uint32_t>();
        return_type_idx = stream.read<std::uint32_t>();
        parameters_off = stream.read<std::uint32_t>();

        DVMType &type = *(dvm_types_pool[return_type_idx].get());
        std::vector<dvmtype_t> params = parse_parameters(stream, parameters_off);

        std::unique_ptr<DVMPrototypeProvider> proto = std::make_unique<DVMPrototypeProvider>(
                string_pool[shorty_idx],
                type,
                params
        );
        std::unique_ptr<DVMPrototype> dvm_proto = std::make_unique<DVMPrototype>(*proto);

        prototypes_pool.emplace_back(std::move(proto));
        dvm_prototype_pool.emplace_back(std::move(dvm_proto));
    }

    stream.seek(current_offset);
    return stream.good();
}

bool
Parser::parse_fields(shuriken::io::ShurikenStream &stream, std::uint32_t fields_offset, std::uint32_t n_of_fields) {
    auto current_offset = stream.position();
    std::uint16_t class_idx, type_idx;
    std::uint32_t name_idx;

    stream.seek(fields_offset);

    for (size_t I = 0; I < n_of_fields; ++I) {
        class_idx = stream.read<std::uint16_t>();
        type_idx = stream.read<std::uint16_t>();
        name_idx = stream.read<std::uint32_t>();

        DVMType &class_ = *(dvm_types_pool[class_idx].get());
        DVMType &type_ = *(dvm_types_pool[type_idx].get());
        std::string_view name_ = string_pool[name_idx];
        fields_.emplace_back(class_, type_, name_);
    }


    stream.seek(current_offset);
    return stream.good();
}

bool
Parser::parse_methods(shuriken::io::ShurikenStream &stream, std::uint32_t methods_offset, std::uint32_t methods_size) {
    auto current_offset = stream.position();
    std::uint16_t class_idx;
    std::uint16_t proto_idx;
    std::uint32_t name_idx;

    stream.seek(methods_offset);

    for (size_t I = 0; I < methods_size; ++I) {
        class_idx = stream.read<std::uint16_t>();
        proto_idx = stream.read<std::uint16_t>();
        name_idx = stream.read<std::uint32_t>();

        DVMType &class_ = *(dvm_types_pool[class_idx].get());
        DVMPrototype &proto_ = *(dvm_prototype_pool[proto_idx].get());
        std::string_view name = string_pool[name_idx];

        methods_.emplace_back(class_, proto_, name);
    }


    stream.seek(current_offset);
    return stream.good();
}

bool
Parser::parse_classes(io::ShurikenStream &stream,
                      std::uint32_t classes_offset,
                      std::uint32_t classes_size) {
    auto current_offset = stream.position();
    stream.seek(classes_offset);

    for (size_t I = 0; I < classes_size; I++) {
        auto class_def = std::make_unique<ClassDef>();
        if (!class_def->parse_class_def(stream,
                                             string_pool,
                                             dvm_types_pool,
                                             fields_,
                                             methods_))
            return false;
        classes_.emplace_back(std::move(class_def));
    }

    stream.seek(current_offset);
    return stream.good();
}

std::vector<std::unique_ptr<DVMTypeProvider>> &Parser::get_types_pool() {
    return types_pool;
}

std::vector<std::unique_ptr<DVMType>> &Parser::get_dvm_types_pool() {
    return dvm_types_pool;
}

std::vector<std::unique_ptr<DVMPrototypeProvider>> &Parser::get_prototypes_pool() {
    return prototypes_pool;
}

std::vector<std::unique_ptr<DVMPrototype>> &Parser::get_dvm_prototype_pool() {
    return dvm_prototype_pool;
}

std::vector<FieldID> &Parser::get_fields_ids() {
    return fields_;
}

std::vector<MethodID> &Parser::get_methods_ids() {
    return methods_;
}

std::vector<std::unique_ptr<ClassDef>> & Parser::get_classes() {
    return classes_;
}
