//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include <functional>
#include <shuriken/sdk/dex/disassembly_constants.hpp>

namespace shuriken {
namespace dex {
class InstructionProvider;
class DalvikIncorrectInstructionProvider;
class Instruction00xProvider;
class Instruction10xProvider;
class Instruction12xProvider;
class Instruction11nProvider;
class Instruction11xProvider;
class Instruction10tProvider;
class Instruction20tProvider;
class Instruction20bcProvider;
class Instruction22xProvider;
class Instruction21tProvider;
class Instruction21sProvider;
class Instruction21hProvider;
class Instruction21cProvider;
class Instruction23xProvider;
class Instruction22bProvider;
class Instruction22tProvider;
class Instruction22sProvider;
class Instruction22cProvider;
class Instruction22csProvider;
class Instruction30tProvider;
class Instruction32xProvider;
class Instruction31iProvider;
class Instruction31tProvider;
class Instruction31cProvider;
class Instruction35cProvider;
class Instruction3rcProvider;
class Instruction45ccProvider;
class Instruction4rccProvider;
class Instruction51lProvider;
class PackedSwitchProvider;
class SparseSwitchProvider;
class FillArrayDataProvivder;

class PackedSwitch;
class SparseSwitch;

using switch_instr_t = std::variant<
        std::monostate,
        PackedSwitch*,
        SparseSwitch*>;

class Instruction {
private:
    std::reference_wrapper<InstructionProvider> instruction;
public:
    // constructors & destructors
    Instruction(InstructionProvider&);
    ~Instruction() = default;

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
private:
    std::reference_wrapper<DalvikIncorrectInstructionProvider> instruction;
public:
    DalvikIncorrectInstruction(DalvikIncorrectInstructionProvider&);
};

class Instruction00x : public Instruction {
private:
    std::reference_wrapper<Instruction00xProvider> instruction;
public:
    Instruction00x(Instruction00xProvider &);
};

class Instruction10x : public Instruction {
private:
    std::reference_wrapper<Instruction10xProvider> instruction;
public:
    Instruction10x(Instruction10xProvider&);
};

class Instruction12x : public Instruction {
private:
    std::reference_wrapper<Instruction12xProvider> instruction;
public:
    Instruction12x(Instruction12xProvider&);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_types() const;
};

class Instruction11n : public Instruction {
private:
    std::reference_wrapper<Instruction11nProvider> instruction;
public:
    Instruction11n(Instruction11nProvider&);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::int8_t getNB() const;
    disassembler::operand_type get_nB_types() const;
};

class Instruction11x : public Instruction {
private:
    std::reference_wrapper<Instruction11xProvider> instruction;
public:
    Instruction11x(Instruction11xProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
};

class Instruction10t : public Instruction {
private:
    std::reference_wrapper<Instruction10tProvider> instruction;
public:
    Instruction10t(Instruction10tProvider&);

    std::int8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;
};

class Instruction20t : public Instruction {
private:
    std::reference_wrapper<Instruction20tProvider> instruction;
public:
    Instruction20t(Instruction20tProvider&);

    std::int16_t getNAAAA() const;
    disassembler::operand_type get_nAAAA_type() const;
};

class Instruction20bc : public Instruction {
private:
    std::reference_wrapper<Instruction20bcProvider> instruction;
public:
    Instruction20bc(Instruction20bcProvider&);

    std::uint8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;
    std::uint16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction22x : public Instruction {
private:
    std::reference_wrapper<Instruction22xProvider> instruction;
public:
    Instruction22x(Instruction22xProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;
};

class Instruction21t : public Instruction {
private:
    std::reference_wrapper<Instruction21tProvider> instruction;
public:
    Instruction21t(Instruction21tProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction21s : public Instruction {
private:
    std::reference_wrapper<Instruction21sProvider> instruction;
public:
    Instruction21s(Instruction21sProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction21h : public Instruction {
private:
    std::reference_wrapper<Instruction21hProvider> instruction;
public:
    Instruction21h(Instruction21hProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int64_t getnBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

class Instruction21c : public Instruction {
private:
    std::reference_wrapper<Instruction21cProvider> instruction;
public:
    Instruction21c(Instruction21cProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint16_t getIBBBB() const;
    disassembler::operand_type get_iBBBB_type() const;
    kind_type_t get_iBBBB_kind();
};

class Instruction23x : public Instruction {
private:
    std::reference_wrapper<Instruction23xProvider> instruction;
public:
    Instruction23x(Instruction23xProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;
    std::uint8_t getVCC() const;
    disassembler::operand_type get_vCC_type() const;
};

class Instruction22b : public Instruction {
private:
    std::reference_wrapper<Instruction22bProvider> instruction;
public:
    Instruction22b(Instruction22bProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;
    std::int8_t getNCC() const;
    disassembler::operand_type get_nCC_type() const;
};

class Instruction22t : public Instruction {
private:
    std::reference_wrapper<Instruction22tProvider> instruction;
public:
    Instruction22t(Instruction22tProvider&);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;
};

class Instruction22s : public Instruction {
private:
    std::reference_wrapper<Instruction22sProvider> instruction;
public:
    Instruction22s(Instruction22sProvider&);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;
};

class Instruction22c : public Instruction {
private:
    std::reference_wrapper<Instruction22cProvider> instruction;
public:
    Instruction22c(Instruction22cProvider&);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;
    kind_type_t get_checked_id_as_kind() const;
};

class Instruction22cs : public Instruction {
private:
    std::reference_wrapper<Instruction22csProvider> instruction;
public:
    Instruction22cs(Instruction22csProvider&);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;
    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;
    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;
    kind_type_t get_field() const;
};

class Instruction30t : public Instruction {
private:
    std::reference_wrapper<Instruction30tProvider> instruction;
public:
    Instruction30t(Instruction30tProvider&);

    std::int32_t getNAAAAAAAA() const;
    disassembler::operand_type get_nAAAAAAAA_type() const;
};

class Instruction32x : public Instruction {
private:
    std::reference_wrapper<Instruction32xProvider> instruction;
public:
    Instruction32x(Instruction32xProvider&);

    std::uint16_t getVAAAA() const;
    disassembler::operand_type get_vAAAA_type() const;
    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;
};

class Instruction31i : public Instruction {
private:
    std::reference_wrapper<Instruction31iProvider> instruction;
public:
    Instruction31i(Instruction31iProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint32_t getNBBBBBBBB() const;
    float getNBBBBBBBB_Float() const;
    disassembler::operand_type get_nBBBBBBBB_type() const;
};

class Instruction31t : public Instruction {
private:
    std::reference_wrapper<Instruction31tProvider> instruction;
public:
    Instruction31t(Instruction31tProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int32_t getNBBBBBBBB() const;
    disassembler::operand_type get_nBBBBBBBB_type() const;
    disassembler::type_of_switch_t get_type_of_switch() const;
    switch_instr_t get_switch() const;
};

class Instruction31c : public Instruction {
private:
    std::reference_wrapper<Instruction31cProvider> instruction;
public:
    Instruction31c(Instruction31cProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint32_t getIBBBBBBBB() const;
    disassembler::operand_type get_IBBBBBBBB_type() const;
    std::string_view get_string_value() const;
    std::string get_string_value_string() const;
};

class Instruction35c : public Instruction {
private:
    std::reference_wrapper<Instruction35cProvider> instruction;
public:
    Instruction35c(Instruction35cProvider&);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint8_t> get_registers();
    disassembler::operand_type get_registers_type() const;
    std::uint16_t get_type_idx() const;
    disassembler::operand_type get_value_type() const;
    disassembler::kind get_value_kind() const;
    kind_type_t get_value() const;
};

class Instruction3rc : public Instruction {
private:
    std::reference_wrapper<Instruction3rcProvider> instruction;
public:
    Instruction3rc(Instruction3rcProvider&);

    std::uint8_t get_registers_size() const;
    std::uint16_t get_index() const;
    kind_type_t get_index_value() const;
    disassembler::operand_type get_index_type() const;
    std::span<std::uint16_t> get_registers();
};

class Instruction45cc : public Instruction {
private:
    std::reference_wrapper<Instruction45ccProvider> instruction;
public:
    Instruction45cc(Instruction45ccProvider&);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint8_t> get_registers();
    std::uint16_t get_method_reference() const;
    kind_type_t get_method_value() const;
    std::uint16_t get_prototype_reference() const;
    kind_type_t get_prototype_value() const;
};

class Instruction4rcc : public Instruction {
private:
    std::reference_wrapper<Instruction4rccProvider> instruction;
public:
    Instruction4rcc(Instruction4rccProvider&);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint16_t> get_registers();
    std::uint16_t get_method_reference() const;
    kind_type_t get_method_value() const;
    std::uint16_t get_prototype_reference() const;
    kind_type_t get_prototype_value() const;
};

class Instruction51l : public Instruction {
private:
    std::reference_wrapper<Instruction51lProvider> instruction;
public:
    Instruction51l(Instruction51lProvider&);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int64_t getNBBBBBBBBBBBBBBBB() const;
    double get_nBBBBBBBBBBBBBBBB_double() const;
    disassembler::operand_type get_nBBBBBBBBBBBBBBBB_type() const;
};

class PackedSwitch : public Instruction {
private:
    std::reference_wrapper<PackedSwitchProvider> instruction;
public:
    PackedSwitch(PackedSwitchProvider&);

    std::uint16_t get_number_of_targets() const;
    std::int32_t get_first_key() const;
    std::span<std::int32_t> get_targets();
};

class SparseSwitch : public Instruction {
private:
    std::reference_wrapper<SparseSwitchProvider> instruction;
public:
    SparseSwitch(SparseSwitchProvider&);

    std::uint16_t get_size_of_targets() const;
    std::span<std::pair<std::int32_t, std::int32_t>> get_keys_targets();
};

class FillArrayData : public Instruction {
private:
    std::reference_wrapper<FillArrayDataProvivder> instruction;
public:
    FillArrayData(FillArrayDataProvivder&);

    std::uint16_t get_element_width() const;
    std::uint32_t get_size_of_data() const;
    std::span<std::uint8_t> get_data();
};

} // namespace dex
} // namespace shuriken