//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/sdk/dex/instruction.hpp>
#include <shuriken/internal/providers/dex/dex_instructions.hpp>

using namespace shuriken::dex;


// Base Instruction class implementation
Instruction::Instruction(InstructionProvider &provider) : instruction(provider) {}

disassembler::kind Instruction::get_kind() const {
    return instruction.get().get_kind();
}

disassembler::dexinsttype Instruction::get_instruction_type() const {
    return instruction.get().get_instruction_type();
}

std::uint32_t Instruction::get_instruction_length() const {
    return instruction.get().get_instruction_length();
}

disassembler::opcodes Instruction::get_instruction_opcode() const {
    return instruction.get().get_instruction_opcode();
}

void Instruction::set_address(std::uint64_t address) {
    instruction.get().set_address(address);
}

std::uint64_t Instruction::get_address() const {
    return instruction.get().get_address();
}

std::span<std::uint8_t> Instruction::get_instruction_bytecode() const {
    return instruction.get().get_instruction_bytecode();
}

std::string_view Instruction::print_instruction() {
    return instruction.get().print_instruction();
}

std::string Instruction::print_instruction_string() {
    return instruction.get().print_instruction_string();
}

bool Instruction::is_terminator() const {
    return instruction.get().is_terminator();
}

bool Instruction::has_side_effects() const {
    return instruction.get().has_side_effects();
}

bool Instruction::may_throw() const {
    return instruction.get().may_throw();
}

bool Instruction::is_instruction_valid() const {
    return instruction.get().is_instruction_valid();
}

const std::string &Instruction::get_error_message() const {
    return instruction.get().get_error_message();
}

// DalvikIncorrectInstruction implementation
DalvikIncorrectInstruction::DalvikIncorrectInstruction(DalvikIncorrectInstructionProvider &provider)
        : Instruction(provider), instruction(provider) {}

// Instruction00x implementation
Instruction00x::Instruction00x(Instruction00xProvider &provider)
        : Instruction(provider), instruction(provider) {}

// Instruction10x implementation
Instruction10x::Instruction10x(Instruction10xProvider &provider)
        : Instruction(provider), instruction(provider) {}

// Instruction12x implementation
Instruction12x::Instruction12x(Instruction12xProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction12x::getVA() const {
    return instruction.get().getVA();
}

disassembler::operand_type Instruction12x::get_vA_type() const {
    return instruction.get().get_vA_type();
}

std::uint8_t Instruction12x::getVB() const {
    return instruction.get().getVB();
}

disassembler::operand_type Instruction12x::get_vB_types() const {
    return instruction.get().get_vB_types();
}

// Instruction11n implementation
Instruction11n::Instruction11n(Instruction11nProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction11n::getVA() const {
    return instruction.get().getVA();
}

disassembler::operand_type Instruction11n::get_vA_type() const {
    return instruction.get().get_vA_type();
}

std::int8_t Instruction11n::getNB() const {
    return instruction.get().getNB();
}

disassembler::operand_type Instruction11n::get_nB_types() const {
    return instruction.get().get_nB_types();
}

// Instruction11x implementation
Instruction11x::Instruction11x(Instruction11xProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction11x::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction11x::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

// Instruction10t implementation
Instruction10t::Instruction10t(Instruction10tProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::int8_t Instruction10t::getNAA() const {
    return instruction.get().getNAA();
}

disassembler::operand_type Instruction10t::get_nAA_type() const {
    return instruction.get().get_nAA_type();
}

// Instruction20t implementation
Instruction20t::Instruction20t(Instruction20tProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::int16_t Instruction20t::getNAAAA() const {
    return instruction.get().getNAAAA();
}

disassembler::operand_type Instruction20t::get_nAAAA_type() const {
    return instruction.get().get_nAAAA_type();
}

// Instruction20bc implementation
Instruction20bc::Instruction20bc(Instruction20bcProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction20bc::getNAA() const {
    return instruction.get().getNAA();
}

disassembler::operand_type Instruction20bc::get_nAA_type() const {
    return instruction.get().get_nAA_type();
}

std::uint16_t Instruction20bc::getNBBBB() const {
    return instruction.get().getNBBBB();
}

disassembler::operand_type Instruction20bc::get_nBBBB_type() const {
    return instruction.get().get_nBBBB_type();
}

// Instruction22x implementation
Instruction22x::Instruction22x(Instruction22xProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction22x::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction22x::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::uint16_t Instruction22x::getVBBBB() const {
    return instruction.get().getVBBBB();
}

disassembler::operand_type Instruction22x::get_vBBBB_type() const {
    return instruction.get().get_vBBBB_type();
}

// Instruction21t implementation
Instruction21t::Instruction21t(Instruction21tProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction21t::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction21t::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::int16_t Instruction21t::getNBBBB() const {
    return instruction.get().getNBBBB();
}

disassembler::operand_type Instruction21t::get_nBBBB_type() const {
    return instruction.get().get_nBBBB_type();
}

// Instruction21s implementation
Instruction21s::Instruction21s(Instruction21sProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction21s::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction21s::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::int16_t Instruction21s::getNBBBB() const {
    return instruction.get().getNBBBB();
}

disassembler::operand_type Instruction21s::get_nBBBB_type() const {
    return instruction.get().get_nBBBB_type();
}

// Instruction21h implementation
Instruction21h::Instruction21h(Instruction21hProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction21h::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction21h::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::int64_t Instruction21h::getnBBBB() const {
    return instruction.get().getnBBBB();
}

disassembler::operand_type Instruction21h::get_nBBBB_type() const {
    return instruction.get().get_nBBBB_type();
}

// Instruction21c implementation
Instruction21c::Instruction21c(Instruction21cProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction21c::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction21c::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::uint16_t Instruction21c::getIBBBB() const {
    return instruction.get().getIBBBB();
}

disassembler::operand_type Instruction21c::get_iBBBB_type() const {
    return instruction.get().get_iBBBB_type();
}

kind_type_t Instruction21c::get_iBBBB_kind() {
    return instruction.get().get_iBBBB_kind();
}

// Instruction23x implementation
Instruction23x::Instruction23x(Instruction23xProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction23x::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction23x::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::uint8_t Instruction23x::getVBB() const {
    return instruction.get().getVBB();
}

disassembler::operand_type Instruction23x::get_vBB_type() const {
    return instruction.get().get_vBB_type();
}

std::uint8_t Instruction23x::getVCC() const {
    return instruction.get().getVCC();
}

disassembler::operand_type Instruction23x::get_vCC_type() const {
    return instruction.get().get_vCC_type();
}

// Instruction22b implementation
Instruction22b::Instruction22b(Instruction22bProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction22b::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction22b::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::uint8_t Instruction22b::getVBB() const {
    return instruction.get().getVBB();
}

disassembler::operand_type Instruction22b::get_vBB_type() const {
    return instruction.get().get_vBB_type();
}

std::int8_t Instruction22b::getNCC() const {
    return instruction.get().getNCC();
}

disassembler::operand_type Instruction22b::get_nCC_type() const {
    return instruction.get().get_nCC_type();
}

// Instruction22t implementation
Instruction22t::Instruction22t(Instruction22tProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction22t::getVA() const {
    return instruction.get().getVA();
}

disassembler::operand_type Instruction22t::get_vA_type() const {
    return instruction.get().get_vA_type();
}

std::uint8_t Instruction22t::getVB() const {
    return instruction.get().getVB();
}

disassembler::operand_type Instruction22t::get_vB_type() const {
    return instruction.get().get_vB_type();
}

std::int16_t Instruction22t::getNCCCC() const {
    return instruction.get().getNCCCC();
}

disassembler::operand_type Instruction22t::get_nCCCC_type() const {
    return instruction.get().get_nCCCC_type();
}

// Instruction22s implementation
Instruction22s::Instruction22s(Instruction22sProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction22s::getVA() const {
    return instruction.get().getVA();
}

disassembler::operand_type Instruction22s::get_vA_type() const {
    return instruction.get().get_vA_type();
}

std::uint8_t Instruction22s::getVB() const {
    return instruction.get().getVB();
}

disassembler::operand_type Instruction22s::get_vB_type() const {
    return instruction.get().get_vB_type();
}

std::int16_t Instruction22s::getNCCCC() const {
    return instruction.get().getNCCCC();
}

disassembler::operand_type Instruction22s::get_nCCCC_type() const {
    return instruction.get().get_nCCCC_type();
}

// Instruction22c implementation
Instruction22c::Instruction22c(Instruction22cProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction22c::getVA() const {
    return instruction.get().getVA();
}

disassembler::operand_type Instruction22c::get_vA_type() const {
    return instruction.get().get_vA_type();
}

std::uint8_t Instruction22c::getVB() const {
    return instruction.get().getVB();
}

disassembler::operand_type Instruction22c::get_vB_type() const {
    return instruction.get().get_vB_type();
}

std::uint16_t Instruction22c::getICCCC() const {
    return instruction.get().getICCCC();
}

disassembler::operand_type Instruction22c::get_iCCCC_type() const {
    return instruction.get().get_iCCCC_type();
}

kind_type_t Instruction22c::get_checked_id_as_kind() const {
    return instruction.get().get_checked_id_as_kind();
}

// Instruction22cs implementation
Instruction22cs::Instruction22cs(Instruction22csProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction22cs::getVA() const {
    return instruction.get().getVA();
}

disassembler::operand_type Instruction22cs::get_vA_type() const {
    return instruction.get().get_vA_type();
}

std::uint8_t Instruction22cs::getVB() const {
    return instruction.get().getVB();
}

disassembler::operand_type Instruction22cs::get_vB_type() const {
    return instruction.get().get_vB_type();
}

std::uint16_t Instruction22cs::getICCCC() const {
    return instruction.get().getICCCC();
}

disassembler::operand_type Instruction22cs::get_iCCCC_type() const {
    return instruction.get().get_iCCCC_type();
}

kind_type_t Instruction22cs::get_field() const {
    return instruction.get().get_field();
}

// Instruction30t implementation
Instruction30t::Instruction30t(Instruction30tProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::int32_t Instruction30t::getNAAAAAAAA() const {
    return instruction.get().getNAAAAAAAA();
}

disassembler::operand_type Instruction30t::get_nAAAAAAAA_type() const {
    return instruction.get().get_nAAAAAAAA_type();
}

// Instruction32x implementation
Instruction32x::Instruction32x(Instruction32xProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint16_t Instruction32x::getVAAAA() const {
    return instruction.get().getVAAAA();
}

disassembler::operand_type Instruction32x::get_vAAAA_type() const {
    return instruction.get().get_vAAAA_type();
}

std::uint16_t Instruction32x::getVBBBB() const {
    return instruction.get().getVBBBB();
}

disassembler::operand_type Instruction32x::get_vBBBB_type() const {
    return instruction.get().get_vBBBB_type();
}

// Instruction31i implementation
Instruction31i::Instruction31i(Instruction31iProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction31i::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction31i::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::uint32_t Instruction31i::getNBBBBBBBB() const {
    return instruction.get().getNBBBBBBBB();
}

float Instruction31i::getNBBBBBBBB_Float() const {
    return instruction.get().getNBBBBBBBB_Float();
}

disassembler::operand_type Instruction31i::get_nBBBBBBBB_type() const {
    return instruction.get().get_nBBBBBBBB_type();
}

// Instruction31t implementation
Instruction31t::Instruction31t(Instruction31tProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction31t::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction31t::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::int32_t Instruction31t::getNBBBBBBBB() const {
    return instruction.get().getNBBBBBBBB();
}

disassembler::operand_type Instruction31t::get_nBBBBBBBB_type() const {
    return instruction.get().get_nBBBBBBBB_type();
}

disassembler::type_of_switch_t Instruction31t::get_type_of_switch() const {
    return instruction.get().get_type_of_switch();
}

switch_instr_t Instruction31t::get_switch() const {
    return instruction.get().get_switch_usr();
}

// Instruction31c implementation
Instruction31c::Instruction31c(Instruction31cProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction31c::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction31c::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::uint32_t Instruction31c::getIBBBBBBBB() const {
    return instruction.get().getIBBBBBBBB();
}

disassembler::operand_type Instruction31c::get_IBBBBBBBB_type() const {
    return instruction.get().get_IBBBBBBBB_type();
}

std::string_view Instruction31c::get_string_value() const {
    return instruction.get().get_string_value();
}

std::string Instruction31c::get_string_value_string() const {
    return instruction.get().get_string_value_string();
}

// Instruction35c implementation
Instruction35c::Instruction35c(Instruction35cProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction35c::get_number_of_registers() const {
    return instruction.get().get_number_of_registers();
}

std::span<std::uint8_t> Instruction35c::get_registers() {
    return instruction.get().get_registers();
}

disassembler::operand_type Instruction35c::get_registers_type() const {
    return instruction.get().get_registers_type();
}

std::uint16_t Instruction35c::get_type_idx() const {
    return instruction.get().get_type_idx();
}

disassembler::operand_type Instruction35c::get_value_type() const {
    return instruction.get().get_value_type();
}

disassembler::kind Instruction35c::get_value_kind() const {
    return instruction.get().get_value_kind();
}

kind_type_t Instruction35c::get_value() const {
    return instruction.get().get_value();
}

// Instruction3rc implementation
Instruction3rc::Instruction3rc(Instruction3rcProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction3rc::get_registers_size() const {
    return instruction.get().get_registers_size();
}

std::uint16_t Instruction3rc::get_index() const {
    return instruction.get().get_index();
}

kind_type_t Instruction3rc::get_index_value() const {
    return instruction.get().get_index_value();
}

disassembler::operand_type Instruction3rc::get_index_type() const {
    return instruction.get().get_index_type();
}

std::span<std::uint16_t> Instruction3rc::get_registers() {
    return instruction.get().get_registers();
}

// Instruction45cc implementation
Instruction45cc::Instruction45cc(Instruction45ccProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction45cc::get_number_of_registers() const {
    return instruction.get().get_number_of_registers();
}

std::span<std::uint8_t> Instruction45cc::get_registers() {
    return instruction.get().get_registers();
}

std::uint16_t Instruction45cc::get_method_reference() const {
    return instruction.get().get_method_reference();
}

kind_type_t Instruction45cc::get_method_value() const {
    return instruction.get().get_method_value();
}

std::uint16_t Instruction45cc::get_prototype_reference() const {
    return instruction.get().get_prototype_reference();
}

kind_type_t Instruction45cc::get_prototype_value() const {
    return instruction.get().get_prototype_value();
}

// Instruction4rcc implementation
Instruction4rcc::Instruction4rcc(Instruction4rccProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction4rcc::get_number_of_registers() const {
    return instruction.get().get_number_of_registers();
}

std::span<std::uint16_t> Instruction4rcc::get_registers() {
    return instruction.get().get_registers();
}

std::uint16_t Instruction4rcc::get_method_reference() const {
    return instruction.get().get_method_reference();
}

kind_type_t Instruction4rcc::get_method_value() const {
    return instruction.get().get_method_value();
}

std::uint16_t Instruction4rcc::get_prototype_reference() const {
    return instruction.get().get_prototype_reference();
}

kind_type_t Instruction4rcc::get_prototype_value() const {
    return instruction.get().get_prototype_value();
}

// Instruction51l implementation
Instruction51l::Instruction51l(Instruction51lProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint8_t Instruction51l::getVAA() const {
    return instruction.get().getVAA();
}

disassembler::operand_type Instruction51l::get_vAA_type() const {
    return instruction.get().get_vAA_type();
}

std::int64_t Instruction51l::getNBBBBBBBBBBBBBBBB() const {
    return instruction.get().getNBBBBBBBBBBBBBBBB();
}

double Instruction51l::get_nBBBBBBBBBBBBBBBB_double() const {
    return instruction.get().get_nBBBBBBBBBBBBBBBB_double();
}

disassembler::operand_type Instruction51l::get_nBBBBBBBBBBBBBBBB_type() const {
    return instruction.get().get_nBBBBBBBBBBBBBBBB_type();
}

// PackedSwitch implementation
PackedSwitch::PackedSwitch(PackedSwitchProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint16_t PackedSwitch::get_number_of_targets() const {
    return instruction.get().get_number_of_targets();
}

std::int32_t PackedSwitch::get_first_key() const {
    return instruction.get().get_first_key();
}

std::span<std::int32_t> PackedSwitch::get_targets() {
    return instruction.get().get_targets();
}

// SparseSwitch implementation
SparseSwitch::SparseSwitch(SparseSwitchProvider &provider)
        : Instruction(provider), instruction(provider) {}

std::uint16_t SparseSwitch::get_size_of_targets() const {
    return instruction.get().get_size_of_targets();
}

std::span<std::pair<std::int32_t, std::int32_t>> SparseSwitch::get_keys_targets() {
    return instruction.get().get_keys_targets();
}

// FillArrayData implementation
FillArrayData::FillArrayData(FillArrayDataProvivder &provider)
        : Instruction(provider), instruction(provider) {}

std::uint16_t FillArrayData::get_element_width() const {
    return instruction.get().get_element_width();
}

std::uint32_t FillArrayData::get_size_of_data() const {
    return instruction.get().get_size_of_data();
}

std::span<std::uint8_t> FillArrayData::get_data() {
    return instruction.get().get_data();
}