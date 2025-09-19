//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include "shuriken/sdk/dex/external_class.hpp"
#include "shuriken/sdk/dex/custom_types.hpp"

#include <algorithm>

namespace shuriken::dex {
class ExternalClass::Impl {
private:
    std::string class_name;
    std::string package_name;
    std::string dalvik_name;
    std::string canonical_name;

    // Xrefs objects
    classxref_t xrefto;

    classxref_t xreffrom;

    std::vector<method_idx_t> xrefnewinstance;

    std::vector<method_idx_t> xrefconstclass;
public:
    Impl(std::string_view dvm_class_name) : dalvik_name(dvm_class_name) {
        // Check if the descriptor is valid (starts with 'L' and ends with ';')
        if (dvm_class_name.empty() || dvm_class_name[0] != 'L' || dvm_class_name.back() != ';') {
            // Invalid format - set empty values
            class_name = "";
            package_name = "";
            canonical_name = "";
            return;
        }

        // Remove the 'L' prefix and ';' suffix to get the type name
        std::string_view type_name = dvm_class_name.substr(1, dvm_class_name.size() - 2);

        // Find the last '/' to separate package from class name
        size_t last_slash = type_name.rfind('/');

        if (last_slash == std::string_view::npos) {
            // No package, just a class name (e.g., "String" -> {"", "String"})
            package_name = "";
            class_name = std::string(type_name);
        } else {
            // Extract package and class name (e.g., "java/lang/String" -> {"java/lang", "String"})
            package_name = std::string(type_name.substr(0, last_slash));
            class_name = std::string(type_name.substr(last_slash + 1));
        }

        // Create canonical name by replacing '/' with '.' in the type name
        canonical_name = std::string(type_name);
        std::replace(canonical_name.begin(), canonical_name.end(), '/', '.');
    }

    /***
     * @return read-only view from class' name
     */
    std::string_view get_name() const {
        return std::string_view{class_name};
    }

    /***
     * @return string with class' name
     */
    std::string get_name_string() const {
        return class_name;
    }

    /**
     * @return name of the package from the class
     */
    std::string_view get_package_name() const {
        return std::string_view{package_name};
    }

    /**
     * @return name of the package as string
     */
    std::string get_package_name_string() const {
        return package_name;
    }

    /**
     * @return name of the class in dalvik format as
     * package/name->className
     */
    std::string_view get_dalvik_name() const {
        return std::string_view{dalvik_name};
    }

    /**
    * @return name of the class in dalvik format as
    * package/name->className as string
    */
    std::string get_dalvik_name_string() const {
        return dalvik_name;
    }

    /**
     * @return name of the class in canonical format as
     * package.name.ClassName
     */
    std::string_view get_canonical_name() const {
        return std::string_view{canonical_name};
    }

    /**
    * @return name of the class in canonical format as
     * package.name.ClassName as string
    */
    std::string get_canonical_name_string() const {
        return canonical_name;
    }

    void add_xref_to(types::ref_type ref_kind,
                     class_external_class_t classobj,
                     method_external_method_t methodobj,
                     std::uint64_t offset) {
        xrefto[classobj].insert(std::make_tuple(ref_kind, methodobj, offset));
    }

    void add_xref_from(types::ref_type ref_kind,
                       class_external_class_t classobj,
                       method_external_method_t methodobj,
                       std::uint64_t offset) {
        xreffrom[classobj].insert(std::make_tuple(ref_kind, methodobj, offset));
    }

    void add_xref_new_instance(method_external_method_t methodobj, std::uint64_t offset) {
        xrefnewinstance.emplace_back(methodobj, offset);
    }

    void add_xref_const_class(method_external_method_t methodobj, std::uint64_t offset) {
        xrefconstclass.emplace_back(methodobj, offset);
    }
};
}