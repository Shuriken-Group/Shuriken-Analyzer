//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file apk_c_core_api.cpp
// @brief Implements the necessary functions for the APK part of
// the core api to work

#include "shuriken/api/C/shuriken_core.h"

#include "shuriken/parser/shuriken_parsers.h"

#include <unordered_map>
#include <vector>

using namespace shuriken::analysis::dex;

namespace {
    const int APK_TAG = 'APKS';

    typedef struct {
        int tag;
        /// @brief APK object with internal data
        std::unique_ptr<shuriken::parser::apk::Apk> apk;
        /// @brief vector of dex files
        std::vector<std::string_view> dex_files;
        /// @brief headers for each DEX file
        std::unordered_map<std::string_view, dexheader_t *> headers_by_dex;
        /// @brief classes for each dex file
        std::unordered_map<std::string_view, std::vector<hdvmclass_t *>> classes_by_dex;
        /// @brief all the methods from the DEX (to access all the methods)
        std::unordered_map<std::string_view, hdvmmethod_t *> methods;

        /// @brief Lazy disassembled methods
        std::unordered_map<std::string_view, dvmdisassembled_method_t *> disassembled_methods;

        /// @brief to check if create the xrefs or not
        bool created_xrefs;
        /// @brief all the class analysis by name
        std::unordered_map<std::string, hdvmclassanalysis_t *> class_analyses;
        /// @brief all the method analysis by name
        std::unordered_map<std::string, hdvmmethodanalysis_t *> method_analyses;
        std::vector<hdvmmethodanalysis_t *> method_analyses_v;
        /// @brief all the field analysis by name
        std::unordered_map<std::string, hdvmfieldanalysis_t *> field_analyses;
        /// @brief all the string analysis
        std::unordered_map<std::string, hdvmstringanalysis_t *> string_analysis;
    } apk_opaque_struct_t;

    hdvmmethod_t *get_method_by_name_from_apk(hApkContext context, const char *method_name) {
        auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
        std::string_view m_name{method_name};
        if (!opaque_struct || opaque_struct->tag != APK_TAG || !opaque_struct->methods.contains(m_name)) return nullptr;
        return opaque_struct->methods.at(m_name);
    }

    namespace fillers {
        /// @brief Add the data to an instruction from the core API, from an instruction from shuriken library
        /// @param instruction instruction from shuriken library.
        /// @param core_instruction instruction from the core API.
        void fill_dex_instruction(shuriken::disassembler::dex::Instruction *instruction,
                                  hdvminstruction_t *core_instruction) {
            core_instruction->instruction_type = static_cast<dexinsttype_e>(instruction->get_instruction_type());
            core_instruction->instruction_length = instruction->get_instruction_length();
            core_instruction->address = instruction->get_address();
            core_instruction->op = instruction->get_instruction_opcode();
            core_instruction->disassembly = instruction->print_instruction().data();
        }

        /// @brief Add the exception information from an exception data of shuriken library, to one from the core C API
        /// @param exception_data exception information from shuriken library.
        /// @param core_exception_data exception information from core API.
        void fill_dex_exception_information(shuriken::disassembler::dex::exception_data_t *exception_data,
                                            dvmexceptions_data_t *core_exception_data) {
            core_exception_data->try_value_start_addr = exception_data->try_value_start_addr;
            core_exception_data->try_value_end_addr = exception_data->try_value_end_addr;
            core_exception_data->n_of_handlers = exception_data->handler.size();
            if (core_exception_data->n_of_handlers > 0) {
                core_exception_data->handler = new dvmhandler_data_t[core_exception_data->n_of_handlers];
                for (size_t i = 0; i < core_exception_data->n_of_handlers; i++) {
                    core_exception_data->handler[i].handler_start_addr = exception_data->handler[i].handler_start_addr;
                    core_exception_data->handler[i].handler_type = exception_data->handler[i].handler_type->get_raw_type().data();
                }
            }
        }

        /// @brief Create a full method from a DisassembledMethod from the shuriken library.
        /// @param opaque_struct opaque structure where the information is stored
        /// @param disassembled_method method to create in the core C API
        /// @param method_core_api method from the core api to fill
        void fill_dex_disassembled_method(
                apk_opaque_struct_t *opaque_struct,
                shuriken::disassembler::dex::DisassembledMethod *disassembled_method,
                dvmdisassembled_method_t *method_core_api) {
            size_t i = 0;
            method_core_api->method_id =
                    get_method_by_name_from_apk(opaque_struct,
                                                disassembled_method->get_method_id()->dalvik_name_format().data());
            method_core_api->n_of_registers = disassembled_method->get_number_of_registers();
            method_core_api->n_of_instructions = disassembled_method->get_number_of_instructions();
            method_core_api->n_of_exceptions = disassembled_method->get_number_of_exceptions();
            method_core_api->method_string = disassembled_method->print_method(true).data();

            // initialize the instructions
            method_core_api->instructions = new hdvminstruction_t[method_core_api->n_of_instructions];

            i = 0;
            for (auto *instruction: disassembled_method->get_instructions()) {
                fill_dex_instruction(instruction, &method_core_api->instructions[i++]);
            }

            // initialize the exception information
            method_core_api->exception_information = new dvmexceptions_data_t[method_core_api->n_of_exceptions];

            i = 0;
            for (auto &exception_data: disassembled_method->get_exceptions()) {
                fill_dex_exception_information(&exception_data, &method_core_api->exception_information[i++]);
            }
        }

        /// @brief Function to create a disassembled method, this is an internal
        /// function
        dvmdisassembled_method_t *return_or_create_disassembled_method_internal(apk_opaque_struct_t *opaque_struct, std::string_view method_name) {
            // if it was previously created
            if (opaque_struct->disassembled_methods.contains(method_name)) return opaque_struct->disassembled_methods[method_name];
            auto *disassembled_method = opaque_struct->apk->get_global_disassembler()->get_disassembled_method(method_name);
            if (disassembled_method == nullptr) return nullptr;
            auto *method = new dvmdisassembled_method_t;
            fill_dex_disassembled_method(opaque_struct, disassembled_method, method);
            // add it to the cache
            opaque_struct->disassembled_methods[disassembled_method->get_method_id()->dalvik_name_format()] = method;
            return method;
        }

        // Methods for the APK Analysis
        /// @brief Create a basic blocks structure given a method, this structure contains all the nodes from the CFG.
        /// for the moment, there are no connections between them
        /// @param opaque_struct opaque structure that contains all the information
        /// @param methodAnalysis method to obtain all the nodes
        /// @return structure which contains all the nodes from the CFG
        basic_blocks_t *create_basic_blocks(apk_opaque_struct_t *opaque_struct, MethodAnalysis *methodAnalysis) {
            basic_blocks_t *bbs = new basic_blocks_t;

            if (methodAnalysis->external()) {
                bbs->n_of_blocks = 0;
                bbs->blocks = nullptr;
                return bbs;
            }

            // get the number of blocks
            bbs->n_of_blocks = methodAnalysis->get_basic_blocks()->get_number_of_basic_blocks();
            // allocate memory for the blocks
            bbs->blocks = new hdvmbasicblock_t[bbs->n_of_blocks];
            // get the disassembled instructions of the method to
            auto *instructions_structure =
                    return_or_create_disassembled_method_internal(opaque_struct, methodAnalysis->get_full_name());
            // now create all the data in the blocks
            int i = 0;
            auto nodes = methodAnalysis->get_basic_blocks()->nodes();
            for (auto *node: nodes) {
                bbs->blocks[i].name = node->get_name().data();
                bbs->blocks[i].try_block = node->is_try_block() ? 1 : 0;
                bbs->blocks[i].catch_block = node->is_catch_block() ? 1 : 0;
                bbs->blocks[i].handler_type = node->is_catch_block() ? node->get_handler_type()->get_raw_type().data() : nullptr;
                bbs->blocks[i].block_string = node->toString().data();
                // For the instructions we will use the disassembled instructions from the disassembler
                bbs->blocks[i].n_of_instructions = node->get_instructions().size();
                std::uint64_t first_instr_addr = node->get_first_address();
                size_t j = 0;
                for (j = 0; j < instructions_structure->n_of_instructions; j++) {
                    if (instructions_structure->instructions[j].address == first_instr_addr) {
                        bbs->blocks[i].instructions = &instructions_structure->instructions[j];
                        break;
                    }
                }
                if (bbs->blocks[i].instructions == nullptr)
                    throw std::runtime_error{"Error, instructions not found in the disassembler."};
                i++;
            }

            return bbs;
        }

        // Methods for the hdvm*

        /// @brief From an EncodedMethod fills the data of a method structure
        /// @param encoded_method method from the internal library
        /// @param method structure for the C core API to fill data
        void fill_dex_method(shuriken::parser::dex::EncodedMethod *encoded_method, hdvmmethod_t *method) {
            auto *const method_id = encoded_method->getMethodID();

            method->class_name = method_id->get_class()->get_raw_type().data();
            method->method_name = method_id->get_method_name().data();
            method->prototype = method_id->get_prototype()->get_dalvik_prototype().data();
            method->access_flags = static_cast<uint16_t>(encoded_method->get_flags());
            if (encoded_method->get_code_item()) {
                method->code_size = static_cast<uint32_t>(encoded_method->get_code_item()->get_bytecode().size());
                method->code = encoded_method->get_code_item()->get_bytecode().data();
            } else {
                method->code_size = 0;
                method->code = nullptr;
            }
            method->dalvik_name = encoded_method->getMethodID()->dalvik_name_format().data();
            method->demangled_name = encoded_method->getMethodID()->demangle().data();
        }

        /// @brief From an EncodedField fills the data of a field structure
        /// @param encoded_field field from the internal library
        /// @param field structure for the C core API to fill data
        void fill_dex_field(shuriken::parser::dex::EncodedField *encoded_field, hdvmfield_t *field) {
            auto *field_id = encoded_field->get_field();
            auto *field_type = field_id->field_type();

            field->class_name = field_id->field_class()->get_raw_type().data();
            field->name = field_id->field_name().data();
            field->access_flags = static_cast<uint16_t>(encoded_field->get_flags());
            field->type_value = field_id->field_type()->get_raw_type().data();
            field->fundamental_value = FUNDAMENTAL_NONE;

            auto type = field_type->get_type();

            if (type == shuriken::parser::dex::FUNDAMENTAL) {
                field->type = FUNDAMENTAL;
                auto *fundamental = dynamic_cast<shuriken::parser::dex::DVMFundamental *>(field_type);
                field->fundamental_value = static_cast<hfundamental_e>(fundamental->get_fundamental_type());
            } else if (type == shuriken::parser::dex::CLASS) {
                field->type = CLASS;
            } else if (type == shuriken::parser::dex::ARRAY) {
                field->type = ARRAY;
                auto *array = reinterpret_cast<shuriken::parser::dex::DVMArray *>(field_type);
                if (array->get_array_base_type()->get_type() == shuriken::parser::dex::FUNDAMENTAL) {
                    const auto *const fundamental = reinterpret_cast<
                            const shuriken::parser::dex::DVMFundamental *>(array->get_array_base_type());
                    field->fundamental_value = static_cast<hfundamental_e>(fundamental->get_fundamental_type());
                }
            } else {
                throw std::runtime_error("Error, not supported type...");
            }
        }

        /// @brief Create a new hdvmclass with the information from the class_def
        /// @param class_def class definition structure with information for the new hdvmclass_t
        /// @param opaque_struct structure with all the metadata from the APK
        /// @return pointer to a newly created hdvmclass_t
        hdvmclass_t *fill_new_hdvmclass_t(shuriken::parser::dex::ClassDef &class_def, apk_opaque_struct_t *opaque_struct) {
            hdvmclass_t *new_class = new hdvmclass_t;
            auto *class_idx = class_def.get_class_idx();
            auto *super_class = class_def.get_superclass();
            auto &class_data_item = class_def.get_class_data_item();

            new_class->class_name = class_idx->get_class_name().data();
            if (super_class)
                new_class->super_class = super_class->get_class_name().data();
            if (!class_def.get_source_file().empty())
                new_class->source_file = class_def.get_source_file().data();

            new_class->access_flags = static_cast<uint16_t>(class_def.get_access_flags());
            new_class->direct_methods_size = static_cast<uint16_t>(class_data_item.get_number_of_direct_methods());
            new_class->virtual_methods_size = static_cast<uint16_t>(class_data_item.get_number_of_virtual_methods());
            new_class->instance_fields_size = static_cast<uint16_t>(class_data_item.get_number_of_instance_fields());
            new_class->static_fields_size = static_cast<uint16_t>(class_data_item.get_number_of_static_fields());

            new_class->virtual_methods = new hdvmmethod_t[new_class->virtual_methods_size];
            for (auto j = 0; j < new_class->virtual_methods_size; j++) {
                fill_dex_method(class_data_item.get_virtual_method_by_id(j), &new_class->virtual_methods[j]);
                opaque_struct->methods.insert({class_data_item.get_virtual_method_by_id(j)->getMethodID()->dalvik_name_format(),
                                               &new_class->virtual_methods[j]});
            }
            new_class->direct_methods = new hdvmmethod_t[new_class->direct_methods_size];
            for (auto j = 0; j < new_class->direct_methods_size; j++) {
                fill_dex_method(class_data_item.get_direct_method_by_id(j), &new_class->direct_methods[j]);
                opaque_struct->methods.insert({class_data_item.get_direct_method_by_id(j)->getMethodID()->dalvik_name_format(),
                                               &new_class->direct_methods[j]});
            }
            /// fill the fields
            new_class->instance_fields = new hdvmfield_t[new_class->instance_fields_size];
            for (auto j = 0; j < new_class->instance_fields_size; j++)
                fill_dex_field(class_data_item.get_instance_field_by_id(j), &new_class->instance_fields[j]);
            new_class->static_fields = new hdvmfield_t[new_class->static_fields_size];
            for (auto j = 0; j < new_class->static_fields_size; j++)
                fill_dex_field(class_data_item.get_static_field_by_id(j), &new_class->static_fields[j]);

            return new_class;
        }

        /// @brief Create all the internal necessary data for the opaque_struct
        void fill_apk_opaque_struct(std::unique_ptr<shuriken::parser::apk::Apk> &apk, apk_opaque_struct_t *opaque_struct) {
            if (opaque_struct == nullptr) return;
            opaque_struct->tag = APK_TAG;
            opaque_struct->apk = std::move(apk);
            auto *working_apk = opaque_struct->apk.get();

            // fill the number and pointer to strings
            auto &dex_files = working_apk->get_dex_files_names();
            for (auto dex_file: dex_files) {
                opaque_struct->dex_files.emplace_back(dex_file);
                opaque_struct->headers_by_dex.insert({dex_file.data(), nullptr});
                auto *parser = working_apk->get_parser_by_file(dex_file.data());
                auto &classes = parser->get_classes();
                // Add the classes from the parser
                for (auto &ref_class_def: classes.get_classdefs_vector()) {
                    auto &class_def = ref_class_def.get();
                    auto *hdvmclass = fill_new_hdvmclass_t(class_def, opaque_struct);
                    opaque_struct->classes_by_dex[dex_file.data()].emplace_back(hdvmclass);
                }
            }
        }

    }// namespace fillers


    // definitions
    namespace getters {
        dexheader_t *get_dex_header(apk_opaque_struct_t *opaque_struct,
                                    const char *dex_file);
        hdvmstringanalysis_t *get_string_analysis(apk_opaque_struct_t *opaque_struct,
                                                  std::string str);
        hdvmfieldanalysis_t *get_field_analysis(apk_opaque_struct_t *opaque_struct,
                                                FieldAnalysis *fieldAnalysis);
        hdvmmethodanalysis_t *get_method_analysis(apk_opaque_struct_t *opaque_struct,
                                                  MethodAnalysis *methodAnalysis);
        hdvmclassanalysis_t *get_class_analysis(apk_opaque_struct_t *opaque_struct,
                                                ClassAnalysis *classAnalysis);

        dexheader_t *get_dex_header(apk_opaque_struct_t *opaque_struct, const char *dex_file) {
            if (opaque_struct->headers_by_dex.contains(dex_file) &&
                    opaque_struct->headers_by_dex.at(dex_file) != nullptr) {
                return opaque_struct->headers_by_dex.at(dex_file);
            }

            auto *working_header = new dexheader_t{};
            auto *parser = opaque_struct->apk->get_parser_by_file(dex_file);
            auto encoded_header = parser->get_header().get_dex_header();

            memcpy(working_header->magic, encoded_header.magic, 8);
            working_header->checksum = encoded_header.checksum;
            memcpy(working_header->signature, encoded_header.signature, 20);
            working_header->file_size = encoded_header.file_size;
            working_header->header_size = encoded_header.header_size;
            working_header->endian_tag = encoded_header.endian_tag;
            working_header->link_size = encoded_header.link_size;
            working_header->link_off = encoded_header.link_off;
            working_header->map_off = encoded_header.map_off;
            working_header->string_ids_size = encoded_header.string_ids_size;
            working_header->string_ids_off = encoded_header.string_ids_off;
            working_header->type_ids_size = encoded_header.type_ids_size;
            working_header->type_ids_off = encoded_header.type_ids_off;
            working_header->proto_ids_size = encoded_header.proto_ids_size;
            working_header->proto_ids_off = encoded_header.proto_ids_off;
            working_header->field_ids_size = encoded_header.field_ids_size;
            working_header->field_ids_off = encoded_header.field_ids_off;
            working_header->method_ids_size = encoded_header.method_ids_size;
            working_header->method_ids_off = encoded_header.method_ids_off;
            working_header->class_defs_size = encoded_header.class_defs_size;
            working_header->class_defs_off = encoded_header.class_defs_off;
            working_header->data_size = encoded_header.data_size;
            working_header->data_off = encoded_header.data_off;

            opaque_struct->headers_by_dex.insert({dex_file, working_header});
            return working_header;
        }


        hdvmstringanalysis_t *get_string_analysis(apk_opaque_struct_t *opaque_struct,
                                                  std::string str) {
            if (opaque_struct->string_analysis.contains(str))
                return opaque_struct->string_analysis[str];
            auto &str_analysis_map = opaque_struct->apk->get_global_analysis()->get_string_analysis();
            StringAnalysis *str_analysis = nullptr;
            for (auto &ref: str_analysis_map) {
                if (str == ref.first) {
                    str_analysis = &ref.second.get();
                    break;
                }
            }
            if (str_analysis == nullptr) return nullptr;
            hdvmstringanalysis_t *f_string = new hdvmstringanalysis_t{};
            opaque_struct->string_analysis.insert({str, f_string});
            f_string->value = reinterpret_cast<const char *>(str_analysis->get_value().data());

            if (opaque_struct->created_xrefs) {
                size_t i = 0;

                // create the xrefs
                auto xrefs = str_analysis->get_xreffrom();
                f_string->n_of_xreffrom = std::distance(xrefs.begin(), xrefs.end());
                f_string->xreffrom = new hdvm_class_method_idx_t[f_string->n_of_xreffrom];
                for (auto xref: xrefs) {
                    f_string->xreffrom[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    f_string->xreffrom[i].method = get_method_analysis(opaque_struct,
                                                                       std::get<MethodAnalysis *>(xref));
                    f_string->xreffrom[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
            }

            return f_string;
        }

        /// @brief get or create a hdvmfieldanalysis_t structure given a FieldAnalysis object
        hdvmfieldanalysis_t *get_field_analysis(apk_opaque_struct_t *opaque_struct,
                                                FieldAnalysis *fieldAnalysis) {
            const auto *full_name = fieldAnalysis->get_name().data();
            auto full_name_str = std::string(full_name);
            if (opaque_struct->field_analyses.contains(full_name_str))
                return opaque_struct->field_analyses[full_name_str];
            hdvmfieldanalysis_t *f_struct = new hdvmfieldanalysis_t{};
            opaque_struct->field_analyses.insert({full_name_str, f_struct});
            f_struct->name = fieldAnalysis->get_name().data();
            if (opaque_struct->created_xrefs) {
                size_t i = 0;

                // Create the xrefread
                auto xrefread = fieldAnalysis->get_xrefread();
                f_struct->n_of_xrefread = std::distance(xrefread.begin(), xrefread.end());
                f_struct->xrefread = new hdvm_class_method_idx_t[f_struct->n_of_xrefread];

                for (auto &xref: xrefread) {
                    f_struct->xrefread[i].cls = get_class_analysis(opaque_struct,
                                                                   std::get<ClassAnalysis *>(xref));
                    f_struct->xrefread[i].method = get_method_analysis(opaque_struct,
                                                                       std::get<MethodAnalysis *>(xref));
                    f_struct->xrefread[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }

                // Create the xrefwrite
                auto xrefwrite = fieldAnalysis->get_xrefwrite();
                f_struct->n_of_xrefwrite = std::distance(xrefwrite.begin(), xrefwrite.end());
                f_struct->xrefwrite = new hdvm_class_method_idx_t[f_struct->n_of_xrefwrite];

                i = 0;
                for (auto &xref: xrefwrite) {
                    f_struct->xrefwrite[i].cls = get_class_analysis(opaque_struct,
                                                                    std::get<ClassAnalysis *>(xref));
                    f_struct->xrefwrite[i].method = get_method_analysis(opaque_struct,
                                                                        std::get<MethodAnalysis *>(xref));
                    f_struct->xrefwrite[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
            }
            return f_struct;
        }

        /// @brief get or create a hdvmmethodanalysis_t structure given a MethodAnalysis object
        hdvmmethodanalysis_t *get_method_analysis(apk_opaque_struct_t *opaque_struct,
                                                  MethodAnalysis *methodAnalysis) {
            auto full_name = methodAnalysis->get_full_name();
            auto full_name_str = std::string(full_name);
            if (opaque_struct->method_analyses.contains(full_name_str))
                return opaque_struct->method_analyses[full_name_str];
            hdvmmethodanalysis_t *method = new hdvmmethodanalysis_t{};
            opaque_struct->method_analyses.insert({full_name_str, method});
            method->name = methodAnalysis->get_name().data();
            method->external = methodAnalysis->external() ? 1 : 0;
            method->is_android_api = methodAnalysis->is_android_api() ? 1 : 0;
            method->descriptor = methodAnalysis->get_descriptor().data();
            method->full_name = methodAnalysis->get_full_name().data();
            method->access_flags = static_cast<access_flags_e>(methodAnalysis->get_access_flags());
            method->class_name = methodAnalysis->get_class_name().data();
            method->method_string = methodAnalysis->toString().data();
            method->basic_blocks = ::fillers::create_basic_blocks(opaque_struct, methodAnalysis);
            if (opaque_struct->created_xrefs) {
                //------------------------------------ xrefread
                auto xrefreads = methodAnalysis->get_xrefread();
                method->n_of_xrefread = std::distance(xrefreads.begin(), xrefreads.end());
                method->xrefread = new hdvm_class_field_idx_t[method->n_of_xrefread];
                int i = 0;
                for (auto &xref: xrefreads) {
                    method->xrefread[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    method->xrefread[i].field = get_field_analysis(opaque_struct, std::get<FieldAnalysis *>(xref));
                    method->xrefread[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
                //------------------------------------ xrefwrite
                auto xrefwrites = methodAnalysis->get_xrefwrite();
                method->n_of_xrefwrite = std::distance(xrefwrites.begin(), xrefwrites.end());
                method->xrefwrite = new hdvm_class_field_idx_t[method->n_of_xrefwrite];
                i = 0;
                for (auto &xref: xrefwrites) {
                    method->xrefwrite[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    method->xrefwrite[i].field = get_field_analysis(opaque_struct, std::get<FieldAnalysis *>(xref));
                    method->xrefwrite[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
                //------------------------------------ xrefto
                auto xrefto = methodAnalysis->get_xrefto();
                method->n_of_xrefto = std::distance(xrefto.begin(), xrefto.end());
                method->xrefto = new hdvm_class_method_idx_t[method->n_of_xrefto];
                i = 0;
                for (auto &xref: xrefto) {
                    method->xrefto[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    method->xrefto[i].method = get_method_analysis(opaque_struct, std::get<MethodAnalysis *>(xref));
                    method->xrefto[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
                //------------------------------------ xreffrom
                auto xreffrom = methodAnalysis->get_xreffrom();
                method->n_of_xreffrom = std::distance(xreffrom.begin(), xreffrom.end());
                method->xreffrom = new hdvm_class_method_idx_t[method->n_of_xreffrom];
                i = 0;
                for (auto &xref: xreffrom) {
                    method->xreffrom[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    method->xreffrom[i].method = get_method_analysis(opaque_struct, std::get<MethodAnalysis *>(xref));
                    method->xreffrom[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
                //------------------------------------ xrefnewinstance
                auto xrefnewinstance = methodAnalysis->get_xrefnewinstance();
                method->n_of_xrefnewinstance = std::distance(xrefnewinstance.begin(), xrefnewinstance.end());
                method->xrefnewinstance = new hdvm_class_idx_t[method->n_of_xrefnewinstance];
                i = 0;
                for (auto &xref: xrefnewinstance) {
                    method->xrefnewinstance[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    method->xrefnewinstance[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
                //------------------------------------ xrefconstclass
                auto xrefconstclass = methodAnalysis->get_xrefconstclass();
                method->n_of_xrefconstclass = std::distance(xrefconstclass.begin(), xrefconstclass.end());
                method->xrefconstclass = new hdvm_class_idx_t[method->n_of_xrefconstclass];
                i = 0;
                for (auto &xref: xrefconstclass) {
                    method->xrefconstclass[i].cls = get_class_analysis(opaque_struct, std::get<ClassAnalysis *>(xref));
                    method->xrefconstclass[i].idx = std::get<std::uint64_t>(xref);
                    i++;
                }
            }
            return method;
        }

        /// @brief get or create a hdvmclassanalysis_t structure given a ClassAnalysis object
        hdvmclassanalysis_t *get_class_analysis(apk_opaque_struct_t *opaque_struct,
                                                ClassAnalysis *classAnalysis) {
            auto full_name = classAnalysis->name();
            auto full_name_str = std::string(full_name);
            if (opaque_struct->class_analyses.contains(full_name_str))
                return opaque_struct->class_analyses[full_name_str];
            hdvmclassanalysis_t *cls = new hdvmclassanalysis_t{};
            opaque_struct->class_analyses.insert({full_name_str, cls});

            cls->is_external = classAnalysis->is_class_external() ? 1 : 0;
            if (!classAnalysis->extends().empty())
                cls->extends_ = classAnalysis->extends().data();
            cls->name_ = classAnalysis->name().data();
            // create the methods
            cls->n_of_methods = classAnalysis->get_nb_methods();
            cls->methods = new hdvmmethodanalysis_t *[cls->n_of_methods];
            auto methods = classAnalysis->get_methods();
            size_t i = 0, j = 0;
            for (auto &method: methods) {
                cls->methods[i++] = get_method_analysis(opaque_struct, method.second);
            }
            // create the fields
            cls->n_of_fields = classAnalysis->get_nb_fields();
            cls->fields = new hdvmfieldanalysis_t *[cls->n_of_fields];
            auto fields = classAnalysis->get_fields();
            i = 0;
            for (auto &fld: fields) {
                cls->fields[i++] = get_field_analysis(opaque_struct,
                                                      fld.second.get());
            }
            if (opaque_struct->created_xrefs) {
                //------------------------ xrefto
                auto xrefto = classAnalysis->get_xrefto();
                cls->n_of_xrefto = std::distance(xrefto.begin(), xrefto.end());
                cls->xrefto = new hdvm_classxref_t[cls->n_of_xrefto];
                i = 0, j = 0;
                for (auto &xref: xrefto) {
                    cls->xrefto[i].classAnalysis = get_class_analysis(opaque_struct, xref.first);
                    auto &set_tuple = xref.second;
                    cls->xrefto[i].n_of_reftype_method_idx = set_tuple.size();
                    cls->xrefto[i].hdvmReftypeMethodIdx = new hdvm_reftype_method_idx_t[cls->xrefto[i].n_of_reftype_method_idx];
                    j = 0;
                    for (auto &tuple: set_tuple) {
                        cls->xrefto[i].hdvmReftypeMethodIdx[j].reType =
                                static_cast<ref_type>(std::get<shuriken::dex::TYPES::ref_type>(tuple));
                        cls->xrefto[i].hdvmReftypeMethodIdx[j].idx =
                                std::get<std::uint64_t>(tuple);
                        cls->xrefto[i].hdvmReftypeMethodIdx[j].methodAnalysis =
                                get_method_analysis(opaque_struct, std::get<MethodAnalysis *>(tuple));
                        j++;
                    }
                    i++;
                }
                //------------------------ xreffrom
                auto xreffrom = classAnalysis->get_xrefsfrom();
                cls->n_of_xreffrom = std::distance(xreffrom.begin(), xreffrom.end());
                cls->xreffrom = new hdvm_classxref_t[cls->n_of_xreffrom];
                i = 0;
                j = 0;
                for (auto &xref: xreffrom) {
                    cls->xreffrom[i].classAnalysis = get_class_analysis(opaque_struct, xref.first);
                    auto &set_tuple = xref.second;
                    cls->xreffrom[i].n_of_reftype_method_idx = set_tuple.size();
                    cls->xreffrom[i].hdvmReftypeMethodIdx = new hdvm_reftype_method_idx_t[cls->xreffrom[i].n_of_reftype_method_idx];
                    j = 0;
                    for (auto &tuple: set_tuple) {
                        cls->xreffrom[i].hdvmReftypeMethodIdx[j].reType =
                                static_cast<ref_type>(std::get<shuriken::dex::TYPES::ref_type>(tuple));
                        cls->xreffrom[i].hdvmReftypeMethodIdx[j].idx =
                                std::get<std::uint64_t>(tuple);
                        cls->xreffrom[i].hdvmReftypeMethodIdx[j].methodAnalysis =
                                get_method_analysis(opaque_struct, std::get<MethodAnalysis *>(tuple));
                        j++;
                    }
                    i++;
                }
                //------------------------ xrefnewinstance
                auto xrefnewinstance = classAnalysis->get_xrefnewinstance();
                cls->n_of_xrefnewinstance = std::distance(xrefnewinstance.begin(), xrefnewinstance.end());
                cls->xrefnewinstance = new hdvm_method_idx_t[cls->n_of_xrefnewinstance];
                i = 0;
                for (auto &xref: xrefnewinstance) {
                    cls->xrefnewinstance[i].idx = std::get<std::uint64_t>(xref);
                    cls->xrefnewinstance[i].method = get_method_analysis(opaque_struct, std::get<MethodAnalysis *>(xref));
                    i++;
                }
                //------------------------ xrefconstclass
                auto xrefconstclass = classAnalysis->get_xrefconstclass();
                cls->n_of_xrefconstclass = std::distance(xrefconstclass.begin(), xrefconstclass.end());
                cls->xrefconstclass = new hdvm_method_idx_t[cls->n_of_xrefconstclass];
                i = 0;
                for (auto &xref: xrefconstclass) {
                    cls->xrefconstclass[i].idx = std::get<std::uint64_t>(xref);
                    cls->xrefconstclass[i].method = get_method_analysis(opaque_struct, std::get<MethodAnalysis *>(xref));
                    i++;
                }
            }
            return cls;
        }

        /// @brief return the vector of all hdvmmethodanalysis_t, or create it and return it
        std::vector<hdvmmethodanalysis_t *> &get_all_method_analysis(apk_opaque_struct_t *opaque_struct) {
            if (!opaque_struct->method_analyses_v.empty()) return opaque_struct->method_analyses_v;
            auto &all_method_analysis = opaque_struct->apk->get_global_analysis()->get_methods();
            for (auto &name_method: all_method_analysis) {
                auto &m_analysis = name_method.second.get();
                auto *hdvmmethodanalysis = get_method_analysis(opaque_struct, &m_analysis);
                opaque_struct->method_analyses_v.emplace_back(hdvmmethodanalysis);
            }
            return opaque_struct->method_analyses_v;
        }
    }// namespace getters

    namespace destroyers {
        void destroy_string_analysis(apk_opaque_struct_t *dex_opaque_struct) {
            for (auto str_str_analysis: dex_opaque_struct->string_analysis) {
                auto *str_analysis = str_str_analysis.second;
                if (dex_opaque_struct->created_xrefs) {
                    delete[] str_analysis->xreffrom;
                }
                delete str_analysis;
                str_analysis = nullptr;
            }
            dex_opaque_struct->string_analysis.clear();
        }

        void destroy_field_analysis(apk_opaque_struct_t *dex_opaque_struct) {
            for (auto &name_field_analysis: dex_opaque_struct->field_analyses) {
                auto *field_analysis = name_field_analysis.second;
                if (dex_opaque_struct->created_xrefs) {
                    delete[] field_analysis->xrefread;
                    delete[] field_analysis->xrefwrite;
                }
                delete field_analysis;
                field_analysis = nullptr;
            }
            dex_opaque_struct->field_analyses.clear();
        }

        void destroy_method_analysis(apk_opaque_struct_t *dex_opaque_struct) {
            for (auto &name_method_analysis: dex_opaque_struct->method_analyses) {
                auto *method_analysis = name_method_analysis.second;
                if (method_analysis->basic_blocks) {
                    if (method_analysis->basic_blocks->blocks) {
                        delete[] method_analysis->basic_blocks->blocks;
                        method_analysis->basic_blocks->blocks = nullptr;
                    }
                    delete method_analysis->basic_blocks;
                    method_analysis->basic_blocks = nullptr;
                }
                if (dex_opaque_struct->created_xrefs) {
                    delete[] method_analysis->xreffrom;
                    delete[] method_analysis->xrefto;
                    delete[] method_analysis->xrefwrite;
                    delete[] method_analysis->xrefread;
                    delete[] method_analysis->xrefconstclass;
                    delete[] method_analysis->xrefnewinstance;
                }
                delete method_analysis;
                method_analysis = nullptr;
            }
            dex_opaque_struct->method_analyses.clear();
        }

        void destroy_class_analysis(apk_opaque_struct_t *dex_opaque_struct) {
            for (auto &name_class_analysis: dex_opaque_struct->class_analyses) {
                auto *class_analysis = name_class_analysis.second;
                delete[] class_analysis->methods;
                delete[] class_analysis->fields;
                if (dex_opaque_struct->created_xrefs) {
                    if (class_analysis->xrefto) {
                        for (size_t i = 0; i < class_analysis->n_of_xrefto; i++) {
                            delete[] class_analysis->xrefto[i].hdvmReftypeMethodIdx;
                            class_analysis->xrefto[i].hdvmReftypeMethodIdx = nullptr;
                        }
                        delete[] class_analysis->xrefto;
                    }
                    if (class_analysis->xreffrom) {
                        for (size_t i = 0; i < class_analysis->n_of_xreffrom; i++) {
                            delete[] class_analysis->xreffrom[i].hdvmReftypeMethodIdx;
                            class_analysis->xreffrom[i].hdvmReftypeMethodIdx = nullptr;
                        }
                        delete[] class_analysis->xreffrom;
                    }

                    if (class_analysis->xrefnewinstance)
                        delete[] class_analysis->xrefnewinstance;

                    if (class_analysis->xrefconstclass)
                        delete[] class_analysis->xrefconstclass;
                }
                delete class_analysis;
                class_analysis = nullptr;
            }
            dex_opaque_struct->class_analyses.clear();
        }

        /// @brief Correctly free the memory from a disassembled method
        /// @param method_core_api method to destroy
        void destroy_disassembled_method(dvmdisassembled_method_t *method_core_api) {
            if (method_core_api->n_of_instructions > 0 && method_core_api->instructions != nullptr)
                delete[] method_core_api->instructions;
            if (method_core_api->n_of_exceptions > 0 && method_core_api->exception_information != nullptr) {
                if (method_core_api->exception_information->n_of_handlers > 0 && method_core_api->exception_information->handler != nullptr) {
                    delete[] method_core_api->exception_information->handler;
                    method_core_api->exception_information->handler = nullptr;
                }
            }
            if (method_core_api->exception_information) {
                delete[] method_core_api->exception_information;
                method_core_api->exception_information = nullptr;
            }
        }

        void destroy_class_data(hdvmclass_t *class_) {
            if (class_->direct_methods) {
                delete[] class_->direct_methods;
                class_->direct_methods = nullptr;
            }
            if (class_->virtual_methods) {
                delete[] class_->virtual_methods;
                class_->virtual_methods = nullptr;
            }
            if (class_->static_fields) {
                delete[] class_->static_fields;
                class_->static_fields = nullptr;
            }
            if (class_->instance_fields) {
                delete[] class_->instance_fields;
                class_->instance_fields = nullptr;
            }
        }

        void destroy_opaque_struct(apk_opaque_struct_t *opaque_struct) {
            // destroy opaque_struct->classes_by_dex
            for (auto &dex_file_classes: opaque_struct->classes_by_dex) {
                for (auto *clazz: dex_file_classes.second) {
                    destroy_class_data(clazz);
                    delete clazz;
                }
            }
            opaque_struct->classes_by_dex.clear();

            // destroy the disassembled methods
            for (auto &method: opaque_struct->disassembled_methods) {
                destroy_disassembled_method(method.second);
                delete method.second;
                method.second = nullptr;
            }

            opaque_struct->disassembled_methods.clear();

            // destroy the class analysis, method analysis and field analysis
            destroy_class_analysis(opaque_struct);
            destroy_method_analysis(opaque_struct);
            destroy_field_analysis(opaque_struct);
            destroy_string_analysis(opaque_struct);

            // finally destroy opaque_struct itself
            delete opaque_struct;
        }
    }// namespace destroyers
}// namespace

///--------------------------- Parser API ---------------------------

hApkContext parse_apk(const char *filePath, boolean_e create_xref) {
    std::unique_ptr<shuriken::parser::apk::Apk> apk;

    try {
        apk = shuriken::parser::parse_apk(filePath, create_xref == FALSE ? false : true);
    } catch (std::runtime_error &) {
        return nullptr;
    }
    auto *opaque_struct = new apk_opaque_struct_t();
    ::fillers::fill_apk_opaque_struct(apk, opaque_struct);
    opaque_struct->created_xrefs = create_xref == 0 ? false : true;
    return reinterpret_cast<hApkContext>(opaque_struct);
}

void destroy_apk(hApkContext context) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return;
    ::destroyers::destroy_opaque_struct(opaque_struct);
}

int get_number_of_dex_files(hApkContext context) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return -1;
    return static_cast<int>(opaque_struct->dex_files.size());
}

const char *get_dex_file_by_index(hApkContext context, unsigned int idx) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    if (idx >= opaque_struct->dex_files.size()) return nullptr;
    return opaque_struct->dex_files[idx].data();
}

dexheader_t *get_header_for_dex_file(hApkContext context, const char *dex_file) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    if (!opaque_struct->headers_by_dex.contains(dex_file)) return nullptr;
    return ::getters::get_dex_header(opaque_struct, dex_file);
}

int get_number_of_classes_for_dex_file(hApkContext context, const char *dex_file) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return -1;
    if (!opaque_struct->classes_by_dex.contains(dex_file)) return -1;
    return static_cast<int>(opaque_struct->classes_by_dex[dex_file].size());
}

hdvmclass_t *get_hdvmclass_from_dex_by_index(hApkContext context, const char *dex_file, unsigned int idx) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    if (!opaque_struct->classes_by_dex.contains(dex_file)) return nullptr;
    if (idx >= opaque_struct->classes_by_dex[dex_file].size()) return nullptr;
    return opaque_struct->classes_by_dex[dex_file][idx];
}

int get_number_of_strings_from_dex(hApkContext context, const char *dex_file) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return -1;
    if (!opaque_struct->classes_by_dex.contains(dex_file)) return -1;
    auto *p = opaque_struct->apk->get_parser_by_file(dex_file);
    auto &strings = p->get_strings();
    return static_cast<int>(strings.get_number_of_strings());
}

const char *get_string_by_id_from_dex(hApkContext context, const char *dex_file, unsigned int i) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    auto *p = opaque_struct->apk->get_parser_by_file(dex_file);
    auto &strings = p->get_strings();
    if (i >= strings.get_number_of_strings()) return nullptr;
    return reinterpret_cast<
            const char *>(strings.get_string_by_id(i).data());
}

//------------------------------------ Disassembly API

dvmdisassembled_method_t *get_disassembled_method_from_apk(hApkContext context, const char *method_name) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    return ::fillers::return_or_create_disassembled_method_internal(opaque_struct, method_name);
}

//------------------------------------- Analysis API

hdvmclassanalysis_t *get_analyzed_class_by_hdvmclass_from_apk(hApkContext context, hdvmclass_t *class_) {
    if (class_ == nullptr) return nullptr;
    return get_analyzed_class_from_apk(context, class_->class_name);
}

hdvmclassanalysis_t *get_analyzed_class_from_apk(hApkContext context, const char *class_name) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    std::string dalvik_name = class_name;
    auto *cls = opaque_struct->apk->get_global_analysis()->get_class_analysis(dalvik_name);
    if (cls == nullptr) return nullptr;
    return ::getters::get_class_analysis(opaque_struct, cls);
}

hdvmmethodanalysis_t *get_analyzed_method_by_hdvmmethod_from_apk(hApkContext context, hdvmmethod_t *method) {
    if (method == nullptr) return nullptr;
    return get_analyzed_method_from_apk(context, method->dalvik_name);
}

hdvmmethodanalysis_t *get_analyzed_method_from_apk(hApkContext context, const char *method_full_name) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    std::string dalvik_name = method_full_name;
    auto *method = opaque_struct->apk->get_global_analysis()->get_method_analysis_by_name(dalvik_name);
    if (method == nullptr) return nullptr;
    return ::getters::get_method_analysis(opaque_struct, method);
}

hdvmstringanalysis_t *get_analyzed_string_from_apk(hApkContext context, const char *string) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    if (string == nullptr) return nullptr;
    std::string cpp_str(string);
    return ::getters::get_string_analysis(opaque_struct, cpp_str);
}

size_t get_number_of_methodanalysis_objects(hApkContext context) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return (size_t) -1;
    size_t value = ::getters::get_all_method_analysis(opaque_struct).size();
    return value;
}

hdvmmethodanalysis_t *get_analyzed_method_by_idx(hApkContext context, size_t idx) {
    auto *opaque_struct = reinterpret_cast<apk_opaque_struct_t *>(context);
    if (!opaque_struct || opaque_struct->tag != APK_TAG) return nullptr;
    auto &all_methods = ::getters::get_all_method_analysis(opaque_struct);
    if (idx >= all_methods.size()) return nullptr;
    return all_methods[idx];
}