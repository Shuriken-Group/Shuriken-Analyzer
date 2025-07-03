//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/disassembly_constants.hpp"
#include "shuriken/internal/engine/dex/dex_engine.hpp"

#include <span>
#include <cstdint>
#include <string>
#include <variant>

namespace shuriken {
namespace dex {

class PackedSwitchProvider;
class SparseSwitchProvider;

using switch_type_t = std::variant<
        std::monostate,
        PackedSwitchProvider*,
        SparseSwitchProvider*>;

class PackedSwitch;
class SparseSwitch;

using switch_type_instr_t = std::variant<
        std::monostate,
        PackedSwitch*,
        SparseSwitch*>;


class InstructionProvider {
private:
    disassembler::dexinsttype instruction_type;
protected:
    std::span<std::uint8_t> op_codes;
    std::uint32_t length;
    disassembler::opcodes opcode;
    std::uint64_t address;
    std::string instruction;

    virtual std::string_view format_instruction() = 0;
    virtual void invalidate_instruction(std::string_view err_msg);

    bool is_valid = true;
    std::string error_message;


public:
    InstructionProvider(std::span<std::uint8_t> bytecode, std::size_t index, disassembler::dexinsttype instruction_type);
    InstructionProvider(std::span<std::uint8_t> bytecode, std::size_t index, disassembler::dexinsttype instruction_type, std::uint32_t length);

    // Virtual functions

    virtual ~InstructionProvider();

    virtual disassembler::kind get_kind() const;

    virtual disassembler::dexinsttype get_instruction_type() const;

    virtual std::uint32_t get_instruction_length() const;

    virtual disassembler::opcodes get_instruction_opcode() const;

    virtual void set_address(std::uint64_t address);

    virtual std::uint64_t get_address() const;

    virtual std::span<std::uint8_t> get_instruction_bytecode() const;

    virtual std::string_view print_instruction() = 0;

    virtual std::string print_instruction_string() = 0;

    virtual bool is_terminator() const;

    virtual bool has_side_effects() const;

    virtual bool may_throw() const;

    virtual bool is_instruction_valid() const;

    virtual const std::string& get_error_message() const;
};

class DalvikIncorrectInstructionProvider : public InstructionProvider {
protected:
    std::string_view format_instruction() override;
public:
    DalvikIncorrectInstructionProvider(std::span<uint8_t> bytecode, std::size_t index,
                                       std::string_view error_message, size_t size_instr,
                                       std::uint64_t address,
                                       disassembler::opcodes opcode);

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction00xProvider : public InstructionProvider {
protected:
    std::string_view format_instruction() override;
public:
    Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction10xProvider : public InstructionProvider {
protected:
    std::string_view format_instruction() override;
public:
    Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction12xProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_types() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction11nProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::int8_t nB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::int8_t getNB() const;
    disassembler::operand_type get_nB_types() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction11xProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
protected:
    std::string_view format_instruction() override;
public:
    Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction10tProvider : public InstructionProvider {
private:
    std::int8_t nAA;
protected:
    std::string_view format_instruction() override;
public:
    Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::int8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction20tProvider : public InstructionProvider {
private:
    std::int16_t nAAAA;
protected:
    std::string_view format_instruction() override;
public:
    Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::int16_t getNAAAA() const;
    disassembler::operand_type get_nAAAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction20bcProvider : public InstructionProvider {
private:
    std::uint8_t nAA;
    std::uint16_t nBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;

    std::uint16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22xProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint16_t vBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21tProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int16_t nBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21sProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int16_t nBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21hProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int64_t nBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int64_t getnBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21cProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint16_t iBBBB;
    kind_type_t source_id;
protected:
    std::string_view format_instruction() override;
public:
    Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint16_t getIBBBB() const;
    disassembler::operand_type get_iBBBB_type() const;

    kind_type_t get_iBBBB_kind();

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction23xProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint8_t vBB;
    std::uint8_t vCC;
protected:
    std::string_view format_instruction() override;
public:
    Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;

    std::uint8_t getVCC() const;
    disassembler::operand_type get_vCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22bProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint8_t vBB;
    std::int8_t nCC;
protected:
    std::string_view format_instruction() override;
public:
    Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;

    std::int8_t getNCC() const;
    disassembler::operand_type get_nCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22tProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::int16_t nCCCC;
protected:
    std::string_view format_instruction() override;
public:
    Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22sProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::int16_t nCCCC;
protected:
    std::string_view format_instruction() override;
public:
    Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22cProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::uint16_t iCCCC;
    kind_type_t checked_id;
protected:
    std::string_view format_instruction() override;
public:
    Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;

    kind_type_t get_checked_id_as_kind() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22csProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::uint16_t iCCCC;
    kind_type_t field;
protected:
    std::string_view format_instruction() override;
public:
    Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;

    kind_type_t get_field() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction30tProvider : public InstructionProvider {
private:
    std::int32_t nAAAAAAAA;
protected:
    std::string_view format_instruction() override;
public:
    Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::int32_t getNAAAAAAAA() const;
    disassembler::operand_type get_nAAAAAAAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction32xProvider : public InstructionProvider {
private:
    std::uint16_t vAAAA;
    std::uint16_t vBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction32xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction32xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint16_t getVAAAA() const;
    disassembler::operand_type get_vAAAA_type() const;

    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction31iProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint32_t nBBBBBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction31iProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction31iProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint32_t getNBBBBBBBB() const;
    float getNBBBBBBBB_Float() const;
    disassembler::operand_type get_nBBBBBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction31tProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int32_t nBBBBBBBB;
    disassembler::type_of_switch_t type_of_switch;
    switch_type_t switch_instruction;
    switch_type_instr_t switch_instruction_usr;
protected:
    std::string_view format_instruction() override;
public:
    Instruction31tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction31tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int32_t getNBBBBBBBB() const;
    disassembler::operand_type get_nBBBBBBBB_type() const;

    disassembler::type_of_switch_t get_type_of_switch() const;

    switch_type_t get_switch() const;

    switch_type_instr_t get_switch_usr() const;

    void set_packed_switch(PackedSwitchProvider * packed_switch);

    void set_sparse_switch(SparseSwitchProvider * sparse_switch);

    void set_packed_switch_usr(PackedSwitch * packed_switch);

    void set_sparse_switch_usr(SparseSwitch * sparse_switch);

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction31cProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint32_t iBBBBBBBB;
    std::string pointed_string;
protected:
    std::string_view format_instruction() override;
public:
    Instruction31cProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction31cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint32_t getIBBBBBBBB() const;
    disassembler::operand_type get_IBBBBBBBB_type() const;

    std::string_view get_string_value() const;
    std::string get_string_value_string() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction35cProvider : public InstructionProvider {
private:
    std::uint8_t array_size;
    std::uint16_t type_index;
    kind_type_t type_value;
    std::vector<std::uint8_t> registers;
protected:
    std::string_view format_instruction() override;
public:
    Instruction35cProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction35cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint8_t> get_registers();
    disassembler::operand_type get_registers_type() const;
    std::uint16_t get_type_idx() const;
    disassembler::operand_type get_value_type() const;
    disassembler::kind get_value_kind() const;
    kind_type_t get_value() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction3rcProvider : public InstructionProvider {
private:
    std::uint8_t array_size;
    std::uint16_t index;
    kind_type_t index_value;
    std::vector<std::uint16_t> registers;
protected:
    std::string_view format_instruction() override;
public:
    Instruction3rcProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction3rcProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t get_registers_size() const;
    std::uint16_t get_index() const;
    kind_type_t get_index_value() const;
    disassembler::operand_type get_index_type() const;
    std::span<std::uint16_t> get_registers();

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction45ccProvider : public InstructionProvider {
private:
    std::uint8_t reg_count;
    std::vector<std::uint8_t> registers;
    std::uint16_t method_reference;
    kind_type_t method_value;
    std::uint16_t prototype_reference;
    kind_type_t prototype_value;
protected:
    std::string_view format_instruction() override;
public:
    Instruction45ccProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction45ccProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint8_t> get_registers();
    std::uint16_t get_method_reference() const;
    kind_type_t get_method_value() const;
    std::uint16_t get_prototype_reference() const;
    kind_type_t get_prototype_value() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction4rccProvider : public InstructionProvider {
private:
    std::uint8_t reg_count;
    std::vector<std::uint16_t> registers;
    std::uint16_t method_reference;
    kind_type_t method_value;
    std::uint16_t prototype_reference;
    kind_type_t prototype_value;
protected:
    std::string_view format_instruction() override;
public:
    Instruction4rccProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction4rccProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t get_number_of_registers() const;
    std::span<std::uint16_t> get_registers();
    std::uint16_t get_method_reference() const;
    kind_type_t get_method_value() const;
    std::uint16_t get_prototype_reference() const;
    kind_type_t get_prototype_value() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction51lProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int64_t nBBBBBBBBBBBBBBBB;
protected:
    std::string_view format_instruction() override;
public:
    Instruction51lProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction51lProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int64_t getNBBBBBBBBBBBBBBBB() const;
    double get_nBBBBBBBBBBBBBBBB_double() const;
    disassembler::operand_type get_nBBBBBBBBBBBBBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class PackedSwitchProvider : public InstructionProvider {
private:
    std::uint16_t size;
    std::int32_t first_key;
    std::vector<std::int32_t> targets;
protected:
    std::string_view format_instruction() override;
public:
    PackedSwitchProvider(std::span<uint8_t> bytecode, std::size_t index);
    PackedSwitchProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint16_t get_number_of_targets() const;
    std::int32_t get_first_key() const;
    std::span<std::int32_t> get_targets();

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class SparseSwitchProvider : public InstructionProvider {
private:
    std::uint16_t size;
    std::vector<std::pair<std::int32_t, std::int32_t>> keys_targets;
protected:
    std::string_view format_instruction() override;
public:
    SparseSwitchProvider(std::span<uint8_t> bytecode, std::size_t index);
    SparseSwitchProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint16_t get_size_of_targets() const;
    std::span<std::pair<std::int32_t, std::int32_t>> get_keys_targets();

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class FillArrayDataProvivder : public InstructionProvider {
private:
    std::uint16_t element_width;
    std::uint32_t size;
    std::vector<std::uint8_t> data;
protected:
    std::string_view format_instruction() override;
public:
    FillArrayDataProvivder(std::span<uint8_t> bytecode, std::size_t index);
    FillArrayDataProvivder(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint16_t get_element_width() const;
    std::uint32_t get_size_of_data() const;
    std::span<std::uint8_t> get_data();

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

} //! namespace dex
} //! namespace shuriken