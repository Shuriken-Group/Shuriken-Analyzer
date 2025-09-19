//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/engine/dex/analysis/xref_generator.hpp"
#include "shuriken/internal/engine/dex/dex_engine.hpp"
#include "shuriken/internal/sdk/dex/method_impl.hpp"
#include "shuriken/internal/sdk/dex/external_method_impl.hpp"
#include "shuriken/internal/sdk/dex/field_impl.hpp"
#include "shuriken/internal/sdk/dex/external_field_impl.hpp"

using namespace shuriken::dex;

void XrefGenerator::analyze_xrefs(DexEngine * dex_engine) {
    this->current_dex_engine = dex_engine;

    for (auto & cls : dex_engine->get_classes()) {
        analyze_class(&cls);
    }
}

void XrefGenerator::analyze_class(Class * current_class) {
    for (auto & method : current_class->get_methods()) {
        analyze_method(&method);
    }
}

void XrefGenerator::analyze_method(Method * method) {
    auto current_cls = &method->get_owner_class();
    auto current_cls_impl = current_dex_engine->get_class_impl_by_class(current_cls);
    auto method_impl = current_dex_engine->get_method_impl_by_method(method);

    for (auto instr : method->get_method_instructions()) {
        auto & instruction = instr.get();
        auto off = instruction.get_address();
        auto op_value = instruction.get_instruction_opcode();

        // check for const-class and new-instance instructions
        if (op_value == disassembler::opcodes::OP_CONST_CLASS ||
            op_value == disassembler::opcodes::OP_NEW_INSTANCE) {
            auto * instr_21c = reinterpret_cast<Instruction21c*>(&instruction);
            DVMType & source_dvmtype = *(std::get<DVMType*>(instr_21c->get_iBBBB_kind()));

            // check we get a TYPE from CONST_CLASS
            // or from NEW_INSTANCE, any other Kind (FIELD, PROTO, etc)
            // it is not valid in this case
            if (instr_21c->get_kind() != disassembler::kind::TYPE ||
                ::get_type(source_dvmtype) != types::type_e::CLASS)
                return;

            auto * dvm_class = std::get<DVMClass*>(source_dvmtype);
            auto cls_name = dvm_class->get_dalvik_format_string();

            // avoid analyzing our own class
            if (cls_name == current_cls_impl->get_dalvik_name_string())
                continue;

            auto cls = current_dex_engine->get_class_by_descriptor(cls_name);
            auto cls_impl = current_dex_engine->get_class_impl_by_class(cls);

            if (cls_impl != nullptr) {
                current_cls_impl->add_xref_to(static_cast<types::ref_type>(op_value), cls, method, off);
                cls_impl->add_xref_from(static_cast<types::ref_type>(op_value), current_cls, method, off);

                if (op_value == disassembler::opcodes::OP_CONST_CLASS) {
                    method_impl->add_xrefconstclass(cls, off);
                    cls_impl->add_xref_const_class(method, off);
                } else {
                    method_impl->add_xrefnewinstance(cls, off);
                    cls_impl->add_xref_new_instance(method, off);
                }
            }
        }
    }
}