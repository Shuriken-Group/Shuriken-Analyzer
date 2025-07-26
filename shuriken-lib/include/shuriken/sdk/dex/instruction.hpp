//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <memory>
#include <shuriken/sdk/dex/disassembly_constants.hpp>

namespace shuriken {
namespace dex {

class PackedSwitch;
class SparseSwitch;

using switch_instr_t = std::variant<
        std::monostate,
        PackedSwitch*,
        SparseSwitch*>;

class Instruction {
protected:
    class Impl;
    std::unique_ptr<Impl> impl;
public:
    // constructors & destructors
    Instruction(Impl*);
    virtual ~Instruction() = default;

    Instruction(const Instruction&) = delete;
    Instruction& operator=(const Instruction&) = delete;

    virtual disassembler::kind get_kind() const;
    virtual disassembler::dexinsttype get_instruction_type() const;
    virtual std::uint32_t get_instruction_length() const;
    virtual disassembler::opcodes get_instruction_opcode() const;
    virtual void set_address(std::uint64_t address);
    virtual std::uint64_t get_address() const;
    virtual std::span<std::uint8_t> get_instruction_bytecode() const;
    virtual std::string_view print_instruction();
    virtual std::string print_instruction_string();
    virtual bool is_terminator() const;
    virtual bool has_side_effects() const;
    virtual bool may_throw() const;
    virtual bool is_instruction_valid() const;
    virtual const std::string& get_error_message() const;
};

class DalvikIncorrectInstruction : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    DalvikIncorrectInstruction(Impl*);
};

class Instruction00x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction00x(Impl*);
};

class Instruction10x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction10x(Impl*);
};

class Instruction12x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction12x(Impl*);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_types() const;
};

class Instruction11n : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction11n(Impl*);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::int8_t getNB() const;
    disassembler::operand_type get_nB_types() const;
};

class Instruction11x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction11x(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
};

class Instruction10t : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction10t(Impl*);

    std::int8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;
};

class Instruction20t : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction20t(Impl*);

    std::int16_t getNAAAA() const;
    disassembler::operand_type get_nAAAA_type() const;
};

class Instruction20bc : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction20bc(Impl*);

    std::uint8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;
    std::uint16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction22x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction22x(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;
};

class Instruction21t : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction21t(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction21s : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction21s(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction21h : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction21h(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int64_t getnBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction21c : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction21c(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint16_t getIBBBB() const;
    disassembler::operand_type get_iBBBB_type() const;
    kind_type_t get_iBBBB_kind();
};

class Instruction23x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction23x(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;
    std::uint8_t getVCC() const;
    disassembler::operand_type get_vCC_type() const;
};

class Instruction22b : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction22b(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;
    std::int8_t getNCC() const;
    disassembler::operand_type get_nCC_type() const;
};

class Instruction22t : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction22t(Impl*);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;
};

class Instruction22s : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction22s(Impl*);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;
};

class Instruction22c : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction22c(Impl*);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;
    kind_type_t get_checked_id_as_kind() const;
};

class Instruction22cs : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction22cs(Impl*);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;
    kind_type_t get_field() const;
};

class Instruction30t : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction30t(Impl*);

    std::int32_t getNAAAAAAAA() const;
    disassembler::operand_type get_nAAAAAAAA_type() const;
};

class Instruction32x : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction32x(Impl*);

    std::uint16_t getVAAAA() const;
    disassembler::operand_type get_vAAAA_type() const;
    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;
};

class Instruction31i : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction31i(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint32_t getNBBBBBBBB() const;
    float getNBBBBBBBB_Float() const;
    disassembler::operand_type get_nBBBBBBBB_type() const;
};

class Instruction31t : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction31t(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int32_t getNBBBBBBBB() const;
    disassembler::operand_type get_nBBBBBBBB_type() const;
    disassembler::type_of_switch_t get_type_of_switch() const;
    switch_instr_t get_switch() const;
    void set_packed_switch(PackedSwitch*);
    void set_sparse_switch(SparseSwitch*);
};

class Instruction31c : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction31c(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint32_t getIBBBBBBBB() const;
    disassembler::operand_type get_IBBBBBBBB_type() const;
    std::string_view get_string_value() const;
    std::string get_string_value_string() const;
};

class Instruction35c : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction35c(Impl*);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint8_t> get_registers();
    disassembler::operand_type get_registers_type() const;
    std::uint16_t get_type_idx() const;
    disassembler::operand_type get_value_type() const;
    disassembler::kind get_value_kind() const;
    kind_type_t get_value() const;
};

class Instruction3rc : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction3rc(Impl*);

    std::uint8_t get_registers_size() const;
    std::uint16_t get_index() const;
    kind_type_t get_index_value() const;
    disassembler::operand_type get_index_type() const;
    std::span<std::uint16_t> get_registers();
};

class Instruction45cc : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction45cc(Impl*);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint8_t> get_registers();
    std::uint16_t get_method_reference() const;
    kind_type_t get_method_value() const;
    std::uint16_t get_prototype_reference() const;
    kind_type_t get_prototype_value() const;
};

class Instruction4rcc : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction4rcc(Impl*);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint16_t> get_registers();
    std::uint16_t get_method_reference() const;
    kind_type_t get_method_value() const;
    std::uint16_t get_prototype_reference() const;
    kind_type_t get_prototype_value() const;
};

class Instruction51l : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    Instruction51l(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int64_t getNBBBBBBBBBBBBBBBB() const;
    double get_nBBBBBBBBBBBBBBBB_double() const;
    disassembler::operand_type get_nBBBBBBBBBBBBBBBB_type() const;
};

class PackedSwitch : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    PackedSwitch(Impl*);

    std::uint16_t get_number_of_targets() const;
    std::int32_t get_first_key() const;
    std::span<std::int32_t> get_targets();
};

class SparseSwitch : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    SparseSwitch(Impl*);

    std::uint16_t get_size_of_targets() const;
    std::span<std::pair<std::int32_t, std::int32_t>> get_keys_targets();
};

class FillArrayData : public Instruction {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    FillArrayData(Impl*);

    std::uint16_t get_element_width() const;
    std::uint32_t get_size_of_data() const;
    std::span<std::uint8_t> get_data();
};

} // namespace dex
} // namespace shuriken