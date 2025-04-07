//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/sdk/dex/dex.hpp>
#include <shuriken/internal/engine/dex/dex_engine.hpp>


using namespace shuriken::dex;

class Dex::Impl {
public:
    std::unique_ptr<DexEngine> dex_engine;
    bool initialized;
    error::Error last_error;
    std::string dex_path_str;

    Impl() : initialized(false), last_error(error::ErrorCode::Success) {}

    ~Impl() = default;
};


shuriken::error::Result<std::unique_ptr<Dex>> shuriken::dex::Dex::create_from_file(std::string_view path) {
    std::unique_ptr<Dex> dex = std::make_unique<Dex>(path);

    if (!dex->initialized()) {
        return shuriken::error::make_error<std::unique_ptr<Dex>>(dex->get_last_error().get_code(),
                                                                 dex->get_last_error().get_message());
    }

    return shuriken::error::make_success(std::move(dex));
}


shuriken::dex::Dex::Dex(std::string_view dex_path) : pimpl(new Dex::Impl()) {
    pimpl->dex_path_str = std::string(dex_path);
    auto stream_result = io::ShurikenStream::from_file(pimpl->dex_path_str);

    if (!stream_result) {
        pimpl->initialized = false;
        pimpl->last_error = stream_result.error();
        return;
    }

    // Simply create the engine - don't use try/catch
    pimpl->dex_engine = std::make_unique<DexEngine>(std::move(stream_result.value()), dex_path, *this);
    auto result = pimpl->dex_engine->parse();
    if (!result) {
        pimpl->initialized = false;
        pimpl->last_error = result.error();
        return;
    }

    pimpl->initialized = true;
}

shuriken::dex::Dex::~Dex() {
    delete pimpl;
}

bool shuriken::dex::Dex::initialized() {
    return pimpl->initialized;
}

shuriken::error::Error shuriken::dex::Dex::get_last_error() {
    return pimpl->last_error;
}

std::string_view shuriken::dex::Dex::get_dex_path() const {
    return pimpl->dex_engine->get_dex_path();
}

std::string shuriken::dex::Dex::get_dex_path_string() const {
    return pimpl->dex_engine->get_dex_path_string();
}

std::string_view shuriken::dex::Dex::get_dex_name() const {
    return pimpl->dex_engine->get_dex_name();
}

std::string shuriken::dex::Dex::get_dex_name_string() const {
    return pimpl->dex_engine->get_dex_name_string();
}

classes_deref_iterator_t shuriken::dex::Dex::get_classes() const {
    return pimpl->dex_engine->get_classes();
}

const Class *
shuriken::dex::Dex::get_class_by_package_name_and_name(std::string_view package_name, std::string_view name) const {
    return pimpl->dex_engine->get_class_by_package_name_and_name(package_name, name);
}

Class *shuriken::dex::Dex::get_class_by_package_name_and_name(std::string_view package_name, std::string_view name) {
    return pimpl->dex_engine->get_class_by_package_name_and_name(package_name, name);
}

const Class *shuriken::dex::Dex::get_class_by_descriptor(std::string_view descriptor) const {
    return pimpl->dex_engine->get_class_by_descriptor(descriptor);
}

Class *shuriken::dex::Dex::get_class_by_descriptor(std::string_view descriptor) {
    return pimpl->dex_engine->get_class_by_descriptor(descriptor);
}

std::vector<Class *> shuriken::dex::Dex::find_classes_by_regex(std::string_view descriptor_regex) {
    return pimpl->dex_engine->find_classes_by_regex(descriptor_regex);
}

method_deref_iterator_t shuriken::dex::Dex::get_methods() const {
    return pimpl->dex_engine->get_methods();
}

const Method *
shuriken::dex::Dex::get_method_by_name_prototype(std::string_view name, std::string_view prototype) const {
    return pimpl->dex_engine->get_method_by_name_prototype(name, prototype);
}

Method *shuriken::dex::Dex::get_method_by_name_prototype(std::string_view name, std::string_view prototype) {
    return pimpl->dex_engine->get_method_by_name_prototype(name, prototype);
}

const Method *shuriken::dex::Dex::get_method_by_descriptor(std::string_view descriptor) const {
    return pimpl->dex_engine->get_method_by_descriptor(descriptor);
}

Method *shuriken::dex::Dex::get_method_by_descriptor(std::string_view descriptor) {
    return pimpl->dex_engine->get_method_by_descriptor(descriptor);
}

fields_deref_iterator_t shuriken::dex::Dex::get_fields() const {
    return pimpl->dex_engine->get_fields();
}

const Field *shuriken::dex::Dex::get_field_by_name(std::string_view name) const {
    return pimpl->dex_engine->get_field_by_name(name);
}

Field *shuriken::dex::Dex::get_field_by_name(std::string_view name) {
    return pimpl->dex_engine->get_field_by_name(name);
}

std::vector<Method *> shuriken::dex::Dex::found_method_by_regex(std::string_view descriptor_regex) {
    return pimpl->dex_engine->found_method_by_regex(descriptor_regex);
}

std::vector<Field *> shuriken::dex::Dex::found_field_by_regex(std::string_view descriptor_regex) {
    return pimpl->dex_engine->found_field_by_regex(descriptor_regex);
}






