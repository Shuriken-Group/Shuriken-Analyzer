//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <functional>
#include <string>
#include <string_view>

#include <shuriken/sdk/common/iterator_range.hpp>

#include <shuriken/sdk/dex/custom_types.hpp>
#include <shuriken/sdk/dex/constants.hpp>

namespace shuriken {
namespace dex {

class DexFieldProvider;
class Dex;
class Class;

class Field {
private:
    std::reference_wrapper<DexFieldProvider> dex_field_provider;
public:
    // constructors & destructors
    Field(DexFieldProvider &);

    ~Field() = default;

    Field(const Field &) = delete;

    Field &operator=(const Field &) = delete;

    // information from the field

    /***
     * @return read-only view from field's name
     */
    std::string_view get_name() const;

    /***
     * @return string with field's name
     */
    std::string get_name_string() const;

    /***
     * @return access flags from the field
     */
    types::access_flags get_field_access_flags() const;

    /***
     * @return const type pointer of the current field
     */
    const DVMType *get_field_type() const;

    /***
     * @return type pointer of the current field
     */
    DVMType *get_field_type();

    /***
     * @return constant pointer to owner class for this Field
     * it can be `nullptr`
     */
    const Class *get_owner_class() const;

    /***
     * @return pointer to owner class for this field
     * it can be `nullptr`
     */
    Class *get_owner_class();

    /***
     * @return constant pointer to dex where the class of this field
     * is
     */
    const Dex *get_owner_dex() const;

    /***
    * @return pointer to dex where the class of this field
    * is
    */
    Dex *get_owner_dex();

    /***
     * @return a view of field's descriptor
     * package_name/class_name->field_name:type
     */
    std::string_view get_descriptor() const;

    /***
     * @return a string of field's descriptor
     * package_name/class_name->field_name:type
     */
    std::string get_descriptor_string() const;

    /***
     * @return iterator to a view of cross-references where
     * the field is being read.
     */
    iterator_range<span_class_method_idx_iterator_t>
    get_xref_read();

    /***
     * @return iterator to a view of cross-references where
     * the field is being written.
     */
    iterator_range<span_class_method_idx_iterator_t>
    get_xref_write();
};

} // namespace dex
} // namespace shuriken