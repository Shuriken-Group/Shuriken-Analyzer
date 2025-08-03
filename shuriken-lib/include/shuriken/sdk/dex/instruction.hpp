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

/**
 * @brief Variant type for switch instruction data
 * 
 * Can contain either PackedSwitch or SparseSwitch instruction data.
 * std::monostate is used when no switch data is present.
 */
using switch_instr_t = std::variant<
        std::monostate,
        PackedSwitch*,
        SparseSwitch*>;

/**
 * @brief Base class for all Dalvik bytecode instructions
 * 
 * This abstract class represents a single Dalvik instruction. It provides
 * common functionality for all instruction types including address tracking,
 * opcode information, bytecode access, and instruction properties.
 * 
 * All specific instruction formats (10x, 11n, 21c, etc.) inherit from this
 * base class and implement their own operand access methods.
 */
class Instruction {
protected:
    class Impl;
    std::unique_ptr<Impl> impl;
public:
    /**
     * @brief Construct a new Instruction object
     * @param impl Pointer to the implementation containing instruction data
     */
    Instruction(Impl*);
    virtual ~Instruction() = default;

    Instruction(const Instruction&) = delete;
    Instruction& operator=(const Instruction&) = delete;

    /**
     * @brief Get the general kind/category of this instruction
     * @return The instruction kind (e.g., NOP, MOVE, RETURN, etc.)
     */
    virtual disassembler::kind get_kind() const;

    /**
     * @brief Get the specific instruction type format
     * @return The instruction format type (e.g., INSTRUCTION_10X, INSTRUCTION_21C, etc.)
     */
    virtual disassembler::dexinsttype get_instruction_type() const;

    virtual disassembler::operation_type get_operation_type() const;

    virtual bool is_jump_instruction() const;

    /**
     * @brief Get the length of this instruction in bytes
     * @return The instruction length in bytes (typically 2, 4, 6, 8, or 10)
     */
    virtual std::uint32_t get_instruction_length() const;

    /**
     * @brief Get the opcode of this instruction
     * @return The specific opcode enum value
     */
    virtual disassembler::opcodes get_instruction_opcode() const;

    /**
     * @brief Set the address/offset of this instruction
     * @param address The bytecode offset where this instruction is located
     */
    virtual void set_address(std::uint64_t address);

    /**
     * @brief Get the address/offset of this instruction
     * @return The bytecode offset where this instruction is located
     */
    virtual std::uint64_t get_address() const;

    /**
     * @brief Get the raw bytecode for this instruction
     * @return Span containing the raw instruction bytes
     */
    virtual std::span<std::uint8_t> get_instruction_bytecode() const;

    /**
     * @brief Get a human-readable string representation of the instruction
     * @return String view of the disassembled instruction
     */
    virtual std::string_view print_instruction();

    /**
     * @brief Get a human-readable string representation as a string copy
     * @return String copy of the disassembled instruction
     */
    virtual std::string print_instruction_string();

    /**
     * @brief Check if this instruction terminates a basic block
     * @return True if this instruction can end control flow (returns, throws, branches)
     */
    virtual bool is_terminator() const;

    /**
     * @brief Check if this instruction has observable side effects
     * @return True if the instruction modifies memory, calls methods, etc.
     */
    virtual bool has_side_effects() const;

    /**
     * @brief Check if this instruction may throw an exception
     * @return True if the instruction can potentially throw an exception
     */
    virtual bool may_throw() const;

    /**
     * @brief Check if this instruction was parsed successfully
     * @return True if the instruction is valid, false if parsing failed
     */
    virtual bool is_instruction_valid() const;

    /**
     * @brief Get error message if instruction parsing failed
     * @return Error message describing what went wrong during parsing
     */
    virtual const std::string& get_error_message() const;
};

// ========================================
// Instruction Format Classes
// ========================================
// 
// Dalvik instructions use different formats identified by their operand layout.
// The format names indicate operand sizes and types:
// - Numbers indicate bit sizes (1=4bits, 2=8bits, 3=16bits, 4=32bits, 5=64bits)
// - Letters indicate operand types:
//   - x: no operands
//   - n: 4-bit literal value  
//   - t: 8/16/32-bit branch target offset
//   - s: 16-bit signed literal value
//   - h: 16-bit high-order bits of 32/64-bit value
//   - c: constant pool index
//   - i: inline cache index
//   - l: 64-bit literal value
//
// Examples:
// - 10x: 1 byte total, 0 operands (NOP)
// - 11n: 1 byte total, 1 register + 1 nibble literal (CONST/4)  
// - 21c: 2 bytes total, 1 register + 1 constant pool index (CONST-STRING)
// - 35c: 3 bytes total, up to 5 registers + 1 constant pool index (INVOKE-*)

/**
 * @brief Represents a malformed or unparseable instruction
 * 
 * This class is used when the disassembler encounters invalid bytecode
 * that cannot be parsed into any known instruction format.
 */
class DalvikIncorrectInstruction : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new DalvikIncorrectInstruction object
     * @param impl Pointer to implementation containing error details
     */
    DalvikIncorrectInstruction(Impl*);
};

/**
 * @brief Format: 00x - 0 operands, 16-bit instruction
 * 
 * Used for simple instructions with no operands.
 * Example: NOP (no operation)
 */
class Instruction00x : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new Instruction00x object
     * @param impl Pointer to implementation containing instruction data
     */
    Instruction00x(Impl*);
};

/**
 * @brief Format: 10x - 0 operands, 16-bit instruction  
 * 
 * Similar to 00x but with different encoding.
 * Example: RETURN-VOID
 */
class Instruction10x : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new Instruction10x object
     * @param impl Pointer to implementation containing instruction data
     */
    Instruction10x(Impl*);
};

/**
 * @brief Format: 12x - 2 registers, 16-bit instruction
 * 
 * Contains two 4-bit register operands (vA and vB).
 * Examples: MOVE vA, vB | ADD-INT vA, vA, vB
 */
class Instruction12x : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new Instruction12x object
     * @param impl Pointer to implementation containing instruction data
     */
    Instruction12x(Impl*);

    /**
     * @brief Get the first register operand (destination)
     * @return 4-bit register number (0-15)
     */
    std::uint8_t getVA() const;

    /**
     * @brief Get the operand type for vA
     * @return Operand type information for the vA register
     */
    disassembler::operand_type get_vA_type() const;

    /**
     * @brief Get the second register operand (source)
     * @return 4-bit register number (0-15)
     */
    std::uint8_t getVB() const;

    /**
     * @brief Get the operand type for vB
     * @return Operand type information for the vB register
     */
    disassembler::operand_type get_vB_types() const;
};

class Instruction11n : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
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
    Impl * impl;
public:
    Instruction11x(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
};

class Instruction10t : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    Instruction10t(Impl*);

    std::int8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;
};

class Instruction20t : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    Instruction20t(Impl*);

    std::int16_t getNAAAA() const;
    disassembler::operand_type get_nAAAA_type() const;
};

class Instruction20bc : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
public:
    Instruction21h(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int64_t getnBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;
};

/**
 * @brief Format: 21c - 1 register + 1 constant pool index, 32-bit instruction
 * 
 * Contains one 8-bit register operand and one 16-bit constant pool index.
 * Used for instructions that reference strings, types, fields, or methods.
 * Examples: CONST-STRING vAA, string@BBBB | NEW-INSTANCE vAA, type@BBBB
 */
class Instruction21c : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new Instruction21c object
     * @param impl Pointer to implementation containing instruction data
     */
    Instruction21c(Impl*);

    /**
     * @brief Get the register operand (destination)
     * @return 8-bit register number
     */
    std::uint8_t getVAA() const;

    /**
     * @brief Get the operand type for vAA register
     * @return Operand type information for the vAA register
     */
    disassembler::operand_type get_vAA_type() const;

    /**
     * @brief Get the constant pool index
     * @return 16-bit index into string, type, field, or method pool
     */
    std::uint16_t getIBBBB() const;

    /**
     * @brief Get the operand type for the constant pool index
     * @return Operand type information for the index
     */
    disassembler::operand_type get_iBBBB_type() const;

    /**
     * @brief Get the resolved constant pool item
     * @return Variant containing the resolved string, type, field, or method
     */
    kind_type_t get_iBBBB_kind();
};

class Instruction23x : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
public:
    Instruction30t(Impl*);

    std::int32_t getNAAAAAAAA() const;
    disassembler::operand_type get_nAAAAAAAA_type() const;
};

class Instruction32x : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
public:
    Instruction31c(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::uint32_t getIBBBBBBBB() const;
    disassembler::operand_type get_IBBBBBBBB_type() const;
    std::string_view get_string_value() const;
    std::string get_string_value_string() const;
};

/**
 * @brief Format: 35c - up to 5 registers + 1 constant pool index, 48-bit instruction
 * 
 * Contains up to 5 register operands (4-bit each) and one 16-bit constant pool index.
 * Primarily used for method invocations with up to 5 arguments.
 * Examples: INVOKE-VIRTUAL {v0, v1, v2}, method@CCCC | INVOKE-STATIC {v0}, method@CCCC
 */
class Instruction35c : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new Instruction35c object
     * @param impl Pointer to implementation containing instruction data
     */
    Instruction35c(Impl*);

    /**
     * @brief Get the number of register arguments (0-5)
     * @return Number of registers used in this instruction
     */
    std::uint8_t get_number_of_registers() const;

    /**
     * @brief Get the register operands
     * @return Span containing the register numbers used as arguments
     */
    std::span<std::uint8_t> get_registers();

    /**
     * @brief Get the operand type for the registers
     * @return Operand type information for the register arguments
     */
    disassembler::operand_type get_registers_type() const;

    /**
     * @brief Get the constant pool index (typically method reference)
     * @return 16-bit index into the method pool
     */
    std::uint16_t get_type_idx() const;

    /**
     * @brief Get the operand type for the constant pool index
     * @return Operand type information for the method reference
     */
    disassembler::operand_type get_value_type() const;

    /**
     * @brief Get the kind of value referenced by the index
     * @return The kind of constant pool item (usually METHOD)
     */
    disassembler::kind get_value_kind() const;

    /**
     * @brief Get the resolved method or other constant pool item
     * @return Variant containing the resolved method reference
     */
    kind_type_t get_value() const;
};

class Instruction3rc : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
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
    Impl * impl;
public:
    Instruction51l(Impl*);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;
    std::int64_t getNBBBBBBBBBBBBBBBB() const;
    double get_nBBBBBBBBBBBBBBBB_double() const;
    disassembler::operand_type get_nBBBBBBBBBBBBBBBB_type() const;
};

/**
 * @brief Packed switch instruction data structure
 * 
 * Contains data for packed-switch instructions where case values are consecutive.
 * More efficient than sparse switch when case values are densely packed.
 * Associated with PACKED-SWITCH-PAYLOAD pseudo-instructions.
 */
class PackedSwitch : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new PackedSwitch object
     * @param impl Pointer to implementation containing switch data
     */
    PackedSwitch(Impl*);

    /**
     * @brief Get the number of switch case targets
     * @return Number of case labels in this switch
     */
    std::uint16_t get_number_of_targets() const;

    /**
     * @brief Get the first case value
     * @return The lowest case value; subsequent cases are consecutive
     */
    std::int32_t get_first_key() const;

    /**
     * @brief Get the branch target offsets for each case
     * @return Span containing relative offsets to case handlers
     */
    std::span<std::int32_t> get_targets();
};

/**
 * @brief Sparse switch instruction data structure
 * 
 * Contains data for sparse-switch instructions where case values can be arbitrary.
 * Used when case values are not consecutive or densely packed.
 * Associated with SPARSE-SWITCH-PAYLOAD pseudo-instructions.
 */
class SparseSwitch : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new SparseSwitch object
     * @param impl Pointer to implementation containing switch data
     */
    SparseSwitch(Impl*);

    /**
     * @brief Get the number of switch case targets
     * @return Number of case labels in this switch
     */
    std::uint16_t get_size_of_targets() const;

    /**
     * @brief Get the case values and their corresponding target offsets
     * @return Span of pairs where first=case value, second=target offset
     */
    std::span<std::pair<std::int32_t, std::int32_t>> get_keys_targets();
};

/**
 * @brief Fill array data instruction payload
 * 
 * Contains initialization data for array fill operations.
 * Associated with FILL-ARRAY-DATA-PAYLOAD pseudo-instructions.
 * Used to efficiently initialize arrays with constant data.
 */
class FillArrayData : public Instruction {
public:
    class Impl;
private:
    Impl * impl;
public:
    /**
     * @brief Construct a new FillArrayData object
     * @param impl Pointer to implementation containing array data
     */
    FillArrayData(Impl*);

    /**
     * @brief Get the width of each array element in bytes
     * @return Element width (1, 2, 4, or 8 bytes)
     */
    std::uint16_t get_element_width() const;

    /**
     * @brief Get the number of elements in the data
     * @return Number of array elements to initialize
     */
    std::uint32_t get_size_of_data() const;

    /**
     * @brief Get the raw initialization data
     * @return Span containing the packed element data
     */
    std::span<std::uint8_t> get_data();
};

} // namespace dex
} // namespace shuriken