//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>


#pragma once

#include <string_view>
#include <string>
#include <memory>

namespace shuriken {
namespace dex {

class ExternalClass {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    ExternalClass(Impl *);
    ~ExternalClass() = default;

    /***
     * @return read-only view from class' name
     */
    std::string_view get_name() const;

    /***
     * @return string with class' name
     */
    std::string get_name_string() const;

    /**
     * @return name of the package from the class
     */
    std::string_view get_package_name() const;

    /**
     * @return name of the package as string
     */
    std::string get_package_name_string() const;

    /**
     * @return name of the class in dalvik format as
     * package/name->className
     */
    std::string_view get_dalvik_name() const;

    /**
    * @return name of the class in dalvik format as
    * package/name->className as string
    */
    std::string get_dalvik_name_string() const;

    /**
     * @return name of the class in canonical format as
     * package.name.ClassName
     */
    std::string_view get_canonical_name() const;

    /**
    * @return name of the class in canonical format as
     * package.name.ClassName as string
    */
    std::string get_canonical_name_string() const;
};

}
}