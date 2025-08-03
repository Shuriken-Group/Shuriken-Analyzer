//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/sdk/dex/instruction.hpp"
#include "shuriken/internal/sdk/dex/instruction_impl.hpp"

#include <memory>

using namespace shuriken::dex;


// Base Instruction class implementation
Instruction::Instruction(Impl *impl) : impl(std::unique_ptr<Impl>(impl)) {}

disassembler::kind Instruction::get_kind() const {
    return impl.get()->get_kind();
}

disassembler::dexinsttype Instruction::get_instruction_type() const {
    return impl.get()->get_instruction_type();
}

disassembler::operation_type Instruction::get_operation_type() const {
    return impl.get()->get_operation_type();
}

bool Instruction::is_jump_instruction() const {
    return impl.get()->is_jump_instruction();
}

std::uint32_t Instruction::get_instruction_length() const {
    return impl.get()->get_instruction_length();
}

disassembler::opcodes Instruction::get_instruction_opcode() const {
    return impl.get()->get_instruction_opcode();
}

void Instruction::set_address(std::uint64_t address) {
    impl.get()->set_address(address);
}

std::uint64_t Instruction::get_address() const {
    return impl.get()->get_address();
}

std::span<std::uint8_t> Instruction::get_instruction_bytecode() const {
    return impl.get()->get_instruction_bytecode();
}

std::string_view Instruction::print_instruction() {
    return impl.get()->print_instruction();
}

std::string Instruction::print_instruction_string() {
    return impl.get()->print_instruction_string();
}

bool Instruction::is_terminator() const {
    return impl.get()->is_terminator();
}

bool Instruction::has_side_effects() const {
    return impl.get()->has_side_effects();
}

bool Instruction::may_throw() const {
    return impl.get()->may_throw();
}

bool Instruction::is_instruction_valid() const {
    return impl.get()->is_instruction_valid();
}

const std::string &Instruction::get_error_message() const {
    return impl.get()->get_error_message();
}

// DalvikIncorrectInstruction implementation
DalvikIncorrectInstruction::DalvikIncorrectInstruction(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

// Instruction00x implementation
Instruction00x::Instruction00x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

// Instruction10x implementation
Instruction10x::Instruction10x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

// Instruction12x implementation
Instruction12x::Instruction12x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction12x::getVA() const {
    return impl->getVA();
}

disassembler::operand_type Instruction12x::get_vA_type() const {
    return impl->get_vA_type();
}

std::uint8_t Instruction12x::getVB() const {
    return impl->getVB();
}

disassembler::operand_type Instruction12x::get_vB_types() const {
    return impl->get_vB_types();
}

// Instruction11n implementation
Instruction11n::Instruction11n(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction11n::getVA() const {
    return impl->getVA();
}

disassembler::operand_type Instruction11n::get_vA_type() const {
    return impl->get_vA_type();
}

std::int8_t Instruction11n::getNB() const {
    return impl->getNB();
}

disassembler::operand_type Instruction11n::get_nB_types() const {
    return impl->get_nB_types();
}

// Instruction11x implementation
Instruction11x::Instruction11x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction11x::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction11x::get_vAA_type() const {
    return impl->get_vAA_type();
}

// Instruction10t implementation
Instruction10t::Instruction10t(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::int8_t Instruction10t::getNAA() const {
    return impl->getNAA();
}

disassembler::operand_type Instruction10t::get_nAA_type() const {
    return impl->get_nAA_type();
}

// Instruction20t implementation
Instruction20t::Instruction20t(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::int16_t Instruction20t::getNAAAA() const {
    return impl->getNAAAA();
}

disassembler::operand_type Instruction20t::get_nAAAA_type() const {
    return impl->get_nAAAA_type();
}

// Instruction20bc implementation
Instruction20bc::Instruction20bc(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction20bc::getNAA() const {
    return impl->getNAA();
}

disassembler::operand_type Instruction20bc::get_nAA_type() const {
    return impl->get_nAA_type();
}

std::uint16_t Instruction20bc::getNBBBB() const {
    return impl->getNBBBB();
}

disassembler::operand_type Instruction20bc::get_nBBBB_type() const {
    return impl->get_nBBBB_type();
}

// Instruction22x implementation
Instruction22x::Instruction22x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction22x::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction22x::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::uint16_t Instruction22x::getVBBBB() const {
    return impl->getVBBBB();
}

disassembler::operand_type Instruction22x::get_vBBBB_type() const {
    return impl->get_vBBBB_type();
}

// Instruction21t implementation
Instruction21t::Instruction21t(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction21t::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction21t::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::int16_t Instruction21t::getNBBBB() const {
    return impl->getNBBBB();
}

disassembler::operand_type Instruction21t::get_nBBBB_type() const {
    return impl->get_nBBBB_type();
}

// Instruction21s implementation
Instruction21s::Instruction21s(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction21s::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction21s::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::int16_t Instruction21s::getNBBBB() const {
    return impl->getNBBBB();
}

disassembler::operand_type Instruction21s::get_nBBBB_type() const {
    return impl->get_nBBBB_type();
}

// Instruction21h implementation
Instruction21h::Instruction21h(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction21h::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction21h::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::int64_t Instruction21h::getnBBBB() const {
    return impl->getnBBBB();
}

disassembler::operand_type Instruction21h::get_nBBBB_type() const {
    return impl->get_nBBBB_type();
}

// Instruction21c implementation
Instruction21c::Instruction21c(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction21c::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction21c::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::uint16_t Instruction21c::getIBBBB() const {
    return impl->getIBBBB();
}

disassembler::operand_type Instruction21c::get_iBBBB_type() const {
    return impl->get_iBBBB_type();
}

kind_type_t Instruction21c::get_iBBBB_kind() {
    return impl->get_iBBBB_kind();
}

// Instruction23x implementation
Instruction23x::Instruction23x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction23x::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction23x::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::uint8_t Instruction23x::getVBB() const {
    return impl->getVBB();
}

disassembler::operand_type Instruction23x::get_vBB_type() const {
    return impl->get_vBB_type();
}

std::uint8_t Instruction23x::getVCC() const {
    return impl->getVCC();
}

disassembler::operand_type Instruction23x::get_vCC_type() const {
    return impl->get_vCC_type();
}

// Instruction22b implementation
Instruction22b::Instruction22b(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction22b::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction22b::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::uint8_t Instruction22b::getVBB() const {
    return impl->getVBB();
}

disassembler::operand_type Instruction22b::get_vBB_type() const {
    return impl->get_vBB_type();
}

std::int8_t Instruction22b::getNCC() const {
    return impl->getNCC();
}

disassembler::operand_type Instruction22b::get_nCC_type() const {
    return impl->get_nCC_type();
}

// Instruction22t implementation
Instruction22t::Instruction22t(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction22t::getVA() const {
    return impl->getVA();
}

disassembler::operand_type Instruction22t::get_vA_type() const {
    return impl->get_vA_type();
}

std::uint8_t Instruction22t::getVB() const {
    return impl->getVB();
}

disassembler::operand_type Instruction22t::get_vB_type() const {
    return impl->get_vB_type();
}

std::int16_t Instruction22t::getNCCCC() const {
    return impl->getNCCCC();
}

disassembler::operand_type Instruction22t::get_nCCCC_type() const {
    return impl->get_nCCCC_type();
}

// Instruction22s implementation
Instruction22s::Instruction22s(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction22s::getVA() const {
    return impl->getVA();
}

disassembler::operand_type Instruction22s::get_vA_type() const {
    return impl->get_vA_type();
}

std::uint8_t Instruction22s::getVB() const {
    return impl->getVB();
}

disassembler::operand_type Instruction22s::get_vB_type() const {
    return impl->get_vB_type();
}

std::int16_t Instruction22s::getNCCCC() const {
    return impl->getNCCCC();
}

disassembler::operand_type Instruction22s::get_nCCCC_type() const {
    return impl->get_nCCCC_type();
}

// Instruction22c implementation
Instruction22c::Instruction22c(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction22c::getVA() const {
    return impl->getVA();
}

disassembler::operand_type Instruction22c::get_vA_type() const {
    return impl->get_vA_type();
}

std::uint8_t Instruction22c::getVB() const {
    return impl->getVB();
}

disassembler::operand_type Instruction22c::get_vB_type() const {
    return impl->get_vB_type();
}

std::uint16_t Instruction22c::getICCCC() const {
    return impl->getICCCC();
}

disassembler::operand_type Instruction22c::get_iCCCC_type() const {
    return impl->get_iCCCC_type();
}

kind_type_t Instruction22c::get_checked_id_as_kind() const {
    return impl->get_checked_id_as_kind();
}

// Instruction22cs implementation
Instruction22cs::Instruction22cs(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction22cs::getVA() const {
    return impl->getVA();
}

disassembler::operand_type Instruction22cs::get_vA_type() const {
    return impl->get_vA_type();
}

std::uint8_t Instruction22cs::getVB() const {
    return impl->getVB();
}

disassembler::operand_type Instruction22cs::get_vB_type() const {
    return impl->get_vB_type();
}

std::uint16_t Instruction22cs::getICCCC() const {
    return impl->getICCCC();
}

disassembler::operand_type Instruction22cs::get_iCCCC_type() const {
    return impl->get_iCCCC_type();
}

kind_type_t Instruction22cs::get_field() const {
    return impl->get_field();
}

// Instruction30t implementation
Instruction30t::Instruction30t(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::int32_t Instruction30t::getNAAAAAAAA() const {
    return impl->getNAAAAAAAA();
}

disassembler::operand_type Instruction30t::get_nAAAAAAAA_type() const {
    return impl->get_nAAAAAAAA_type();
}

// Instruction32x implementation
Instruction32x::Instruction32x(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint16_t Instruction32x::getVAAAA() const {
    return impl->getVAAAA();
}

disassembler::operand_type Instruction32x::get_vAAAA_type() const {
    return impl->get_vAAAA_type();
}

std::uint16_t Instruction32x::getVBBBB() const {
    return impl->getVBBBB();
}

disassembler::operand_type Instruction32x::get_vBBBB_type() const {
    return impl->get_vBBBB_type();
}

// Instruction31i implementation
Instruction31i::Instruction31i(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction31i::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction31i::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::uint32_t Instruction31i::getNBBBBBBBB() const {
    return impl->getNBBBBBBBB();
}

float Instruction31i::getNBBBBBBBB_Float() const {
    return impl->getNBBBBBBBB_Float();
}

disassembler::operand_type Instruction31i::get_nBBBBBBBB_type() const {
    return impl->get_nBBBBBBBB_type();
}

// Instruction31t implementation
Instruction31t::Instruction31t(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction31t::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction31t::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::int32_t Instruction31t::getNBBBBBBBB() const {
    return impl->getNBBBBBBBB();
}

disassembler::operand_type Instruction31t::get_nBBBBBBBB_type() const {
    return impl->get_nBBBBBBBB_type();
}

disassembler::type_of_switch_t Instruction31t::get_type_of_switch() const {
    return impl->get_type_of_switch();
}

switch_instr_t Instruction31t::get_switch() const {
    return impl->get_switch();
}

void Instruction31t::set_packed_switch(PackedSwitch *packedSwitch) {
    impl->set_packed_switch(packedSwitch);
}


void Instruction31t::set_sparse_switch(SparseSwitch *sparseSwitch) {
    impl->set_sparse_switch(sparseSwitch);
}

// Instruction31c implementation
Instruction31c::Instruction31c(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction31c::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction31c::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::uint32_t Instruction31c::getIBBBBBBBB() const {
    return impl->getIBBBBBBBB();
}

disassembler::operand_type Instruction31c::get_IBBBBBBBB_type() const {
    return impl->get_IBBBBBBBB_type();
}

std::string_view Instruction31c::get_string_value() const {
    return impl->get_string_value();
}

std::string Instruction31c::get_string_value_string() const {
    return impl->get_string_value_string();
}

// Instruction35c implementation
Instruction35c::Instruction35c(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction35c::get_number_of_registers() const {
    return impl->get_number_of_registers();
}

std::span<std::uint8_t> Instruction35c::get_registers() {
    return impl->get_registers();
}

disassembler::operand_type Instruction35c::get_registers_type() const {
    return impl->get_registers_type();
}

std::uint16_t Instruction35c::get_type_idx() const {
    return impl->get_type_idx();
}

disassembler::operand_type Instruction35c::get_value_type() const {
    return impl->get_value_type();
}

disassembler::kind Instruction35c::get_value_kind() const {
    return impl->get_value_kind();
}

kind_type_t Instruction35c::get_value() const {
    return impl->get_value();
}

// Instruction3rc implementation
Instruction3rc::Instruction3rc(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction3rc::get_registers_size() const {
    return impl->get_registers_size();
}

std::uint16_t Instruction3rc::get_index() const {
    return impl->get_index();
}

kind_type_t Instruction3rc::get_index_value() const {
    return impl->get_index_value();
}

disassembler::operand_type Instruction3rc::get_index_type() const {
    return impl->get_index_type();
}

std::span<std::uint16_t> Instruction3rc::get_registers() {
    return impl->get_registers();
}

// Instruction45cc implementation
Instruction45cc::Instruction45cc(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction45cc::get_number_of_registers() const {
    return impl->get_number_of_registers();
}

std::span<std::uint8_t> Instruction45cc::get_registers() {
    return impl->get_registers();
}

std::uint16_t Instruction45cc::get_method_reference() const {
    return impl->get_method_reference();
}

kind_type_t Instruction45cc::get_method_value() const {
    return impl->get_method_value();
}

std::uint16_t Instruction45cc::get_prototype_reference() const {
    return impl->get_prototype_reference();
}

kind_type_t Instruction45cc::get_prototype_value() const {
    return impl->get_prototype_value();
}

// Instruction4rcc implementation
Instruction4rcc::Instruction4rcc(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction4rcc::get_number_of_registers() const {
    return impl->get_number_of_registers();
}

std::span<std::uint16_t> Instruction4rcc::get_registers() {
    return impl->get_registers();
}

std::uint16_t Instruction4rcc::get_method_reference() const {
    return impl->get_method_reference();
}

kind_type_t Instruction4rcc::get_method_value() const {
    return impl->get_method_value();
}

std::uint16_t Instruction4rcc::get_prototype_reference() const {
    return impl->get_prototype_reference();
}

kind_type_t Instruction4rcc::get_prototype_value() const {
    return impl->get_prototype_value();
}

// Instruction51l implementation
Instruction51l::Instruction51l(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint8_t Instruction51l::getVAA() const {
    return impl->getVAA();
}

disassembler::operand_type Instruction51l::get_vAA_type() const {
    return impl->get_vAA_type();
}

std::int64_t Instruction51l::getNBBBBBBBBBBBBBBBB() const {
    return impl->getNBBBBBBBBBBBBBBBB();
}

double Instruction51l::get_nBBBBBBBBBBBBBBBB_double() const {
    return impl->get_nBBBBBBBBBBBBBBBB_double();
}

disassembler::operand_type Instruction51l::get_nBBBBBBBBBBBBBBBB_type() const {
    return impl->get_nBBBBBBBBBBBBBBBB_type();
}

// PackedSwitch implementation
PackedSwitch::PackedSwitch(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint16_t PackedSwitch::get_number_of_targets() const {
    return impl->get_number_of_targets();
}

std::int32_t PackedSwitch::get_first_key() const {
    return impl->get_first_key();
}

std::span<std::int32_t> PackedSwitch::get_targets() {
    return impl->get_targets();
}

// SparseSwitch implementation
SparseSwitch::SparseSwitch(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint16_t SparseSwitch::get_size_of_targets() const {
    return impl->get_size_of_targets();
}

std::span<std::pair<std::int32_t, std::int32_t>> SparseSwitch::get_keys_targets() {
    return impl->get_keys_targets();
}

// FillArrayData implementation
FillArrayData::FillArrayData(Impl *impl)
        : Instruction(impl) { this->impl = impl; }

std::uint16_t FillArrayData::get_element_width() const {
    return impl->get_element_width();
}

std::uint32_t FillArrayData::get_size_of_data() const {
    return impl->get_size_of_data();
}

std::span<std::uint8_t> FillArrayData::get_data() {
    return impl->get_data();
}