//
// Created by fare9 on 30/12/23.
//

#include <chrono>
#include <fmt/core.h>
#include <functional>
#include <iostream>
// Dex & APK stuff
#include <shuriken/analysis/Dex/analysis.h>
#include <shuriken/common/Dex/dvm_types.h>
#include <shuriken/disassembler/Dex/dex_disassembler.h>
#include <shuriken/parser/shuriken_parsers.h>


#include <vector>

void show_help(std::string &prog_name) {
    fmt::println("USAGE: {} <dex/apk file to analyze> [-h] [-c] [-f] [-m] [-b]", prog_name);
    fmt::println(" -h: show file header");
    fmt::println(" -c: show classes from file");
    fmt::println(" -f: show fields from classes (it needs -c)");
    fmt::println(" -m: show methods from classes (it needs -c)");
    fmt::println(" -b: show bytecode from methods (it needs -m)");
    fmt::println(" -D: show the disassembled code from methods (it needs -m)");
    fmt::println(" -B: show the methods as basic blocks (it needs -m)");
    fmt::println(" -x: show the xrefs from classes (it needs -c), from methods (it requires -m) or from fields (it needs -f)");
    fmt::println(" -T: measure and print after the execution the time taken for the analysis");
    fmt::println(" -N: analyze but do not print any information");
    #ifdef MACHO
        fmt::println(" -mh: show Mach-O file header");
        fmt::println(" -mlcs: show Mach-O file load commands");
        fmt::println(" -msegs: show Mach-O file segment commands");
        fmt::println(" -msecs: show Mach-O file sections");
    #endif
}

void parse_dex(std::string& dex_file);
void parse_apk(std::string& apk_file);
void print_header(shuriken::parser::dex::DexHeader &);
void print_classes(shuriken::parser::dex::DexClasses &);
void print_method(shuriken::parser::dex::EncodedMethod *, size_t);
void print_field(shuriken::parser::dex::EncodedField *, size_t);
void print_code(std::span<std::uint8_t>);
#ifdef MACHO
    void parse_macho(std::string& macho_file);
    void print_macho_header(const shuriken::parser::macho::MachoHeader &macho_header);
    void print_macho_loadcommands(const shuriken::parser::macho::MachoCommands &commands);
    void print_macho_segmentcommands(const shuriken::parser::macho::MachoCommands &commands);
    void print_macho_sections(const shuriken::parser::macho::MachoSections &sections);
#endif 

bool headers = false;
bool show_classes = false;
bool methods = false;
bool fields = false;
bool code = false;
bool disassembly = false;
bool blocks = false;
bool running_time = false;
bool xrefs = false;
bool no_print = false;
#ifdef MACHO
    bool macho_headers = false;
    bool macho_loadcommands = false;
    bool macho_segmentcommands = false;
    bool macho_sections = false;
#endif

std::unique_ptr<shuriken::parser::apk::Apk> parsed_apk = nullptr;
std::unique_ptr<shuriken::parser::dex::Parser> parsed_dex = nullptr;
std::unique_ptr<shuriken::disassembler::dex::DexDisassembler> disassembler_own = nullptr;
shuriken::disassembler::dex::DexDisassembler * disassembler = nullptr;
std::unique_ptr<shuriken::analysis::dex::Analysis> dex_analysis_own = nullptr;
shuriken::analysis::dex::Analysis * analysis = nullptr;
#ifdef MACHO
    std::unique_ptr<shuriken::parser::macho::Parser> parsed_macho = nullptr;
#endif

int main(int argc, char **argv) {
    std::vector<std::string> args{argv, argv + argc};

    auto start_time = std::chrono::high_resolution_clock::now();

    if (args.size() == 1) {
        show_help(args[0]);
        return -1;
    }

    std::unordered_map<std::string, std::function<void()>> options{
            {"-h", [&]() { headers = true; }},
            {"-c", [&]() { show_classes = true; }},
            {"-m", [&]() { methods = true; }},
            {"-f", [&]() { fields = true; }},
            {"-b", [&]() { code = true; }},
            {"-D", [&]() { disassembly = true; }},
            {"-B", [&]() { blocks = true; }},
            {"-x", [&]() { xrefs = true; }},
            {"-T", [&]() { running_time = true; }},
            {"-N", [&]() { no_print = true; }},
            #ifdef MACHO
                {"-mh", [&]() { macho_headers = true; }},
                {"-mlcs", [&]() { macho_loadcommands = true; }},
                {"-msegs", [&]() { macho_segmentcommands = true; }},
                {"-msecs", [&]() { macho_sections = true; }}
            #endif
    };

    for (const auto &s: args) {
        if (auto it = options.find(s); it != options.end()) {
            it->second();
        }
    }

    try {
        if (args[1].ends_with(".dex")) { // manage dex file :)
            parse_dex(args[1]);
        } else if (args[1].ends_with(".apk")) {
            parse_apk(args[1]);
        }
        #ifdef MACHO
            else {
                parse_macho(args[1]);
            }
        #endif
    } catch (std::runtime_error &re) {
        fmt::println("Exception: {}", re.what());
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    if (running_time) {
        // Calculate the duration
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Convert duration to hours, minutes, seconds, and milliseconds
        auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

        // Print the duration
        fmt::print("Execution time: {:02}h:{:02}m:{:02}s:{:03}ms\n",
                   hours.count(), minutes.count(), seconds.count(), milliseconds.count());
    }

    return 0;
}

void parse_dex(std::string& dex_file) {
    parsed_dex = shuriken::parser::parse_dex(dex_file);

    if (disassembly) {
        disassembler_own = std::make_unique<shuriken::disassembler::dex::DexDisassembler>(parsed_dex.get());
        disassembler_own->disassembly_dex();
        disassembler = disassembler_own.get();
    }

    if (blocks || xrefs) {
        if (disassembler_own == nullptr) {
            disassembler_own = std::make_unique<shuriken::disassembler::dex::DexDisassembler>(parsed_dex.get());
            disassembler_own->disassembly_dex();
        }
        dex_analysis_own = std::make_unique<shuriken::analysis::dex::Analysis>(parsed_dex.get(),
                                                                               disassembler_own.get(),
                                                                               xrefs ? true : false);
        dex_analysis_own->create_xrefs();

        disassembler = disassembler_own.get();
        analysis = dex_analysis_own.get();
    }

    if (!no_print) {
        auto &header = parsed_dex->get_header();

        if (headers) print_header(header);
        if (show_classes) print_classes(parsed_dex->get_classes());
    }
}

void parse_apk(std::string& apk_file) {
    parsed_apk = shuriken::parser::parse_apk(apk_file, xrefs ? true : false);

    disassembler = parsed_apk->get_global_disassembler();
    analysis = parsed_apk->get_global_analysis();

    for (auto & file_name : parsed_apk->get_dex_files_names()) {
        fmt::println("DEX File: {}", file_name);
    }

    if (!no_print) {
        for (auto & file_parser : parsed_apk->get_dex_parsers()) {
            auto & parsed_dex = file_parser.second.get();
            auto & header = parsed_dex.get_header();

            fmt::println("Analysis of file: {}", file_parser.first);
            if (headers) print_header(header);
            if (show_classes) print_classes(parsed_dex.get_classes());
        }
    }

}

void print_header(shuriken::parser::dex::DexHeader &header) {
    auto &dex_header = header.get_dex_header();
    fmt::println("Dex Header:");
    fmt::print(" Magic:");
    for (auto b: dex_header.magic) {
        if (isprint(b))
            fmt::print(" {:02X} ({:c})", b, b);
        else
            fmt::print(" {:02X} ( )", b, b);
    }
    fmt::println("\n Checksum:              0x{:X}", static_cast<std::uint32_t>(dex_header.checksum));
    fmt::print(" Signature:");
    for (auto b: dex_header.signature) {
        fmt::print(" {:02X}", b);
    }
    fmt::print("\n File Size:             {}\n", dex_header.file_size);
    fmt::print(" Header Size:           {}\n", dex_header.header_size);
    fmt::print(" Endian Tag:            0x{:X}\n", dex_header.endian_tag);
    fmt::print(" Link offset:           0x{:X}\n", dex_header.link_off);
    fmt::print(" Link Size:             {}\n", dex_header.link_size);
    fmt::print(" Map offset:            0x{:X}\n", dex_header.map_off);
    fmt::print(" String ids offset:     0x{:X}\n", dex_header.string_ids_off);
    fmt::print(" String ids size:       {}\n", dex_header.string_ids_size);
    fmt::print(" Type ids offset:       0x{:X}\n", dex_header.type_ids_off);
    fmt::print(" Type ids size:         {}\n", dex_header.type_ids_size);
    fmt::print(" Proto ids offset:      0x{:X}\n", dex_header.proto_ids_off);
    fmt::print(" Proto ids size:        {}\n", dex_header.proto_ids_size);
    fmt::print(" Field ids offset:      0x{:X}\n", dex_header.field_ids_off);
    fmt::print(" Field ids size:        {}\n", dex_header.field_ids_size);
    fmt::print(" Method ids offset:     0x{:X}\n", dex_header.method_ids_off);
    fmt::print(" Method ids size:       {}\n", dex_header.method_ids_size);
    fmt::print(" Class ids offset:      0x{:X}\n", dex_header.class_defs_off);
    fmt::print(" Class ids size:        {}\n", dex_header.class_defs_size);
    fmt::print(" Data ids offset:       0x{:X}\n", dex_header.data_off);
    fmt::print(" Data ids size:         {}\n", dex_header.data_size);
}

void print_classes(shuriken::parser::dex::DexClasses &classes) {
    size_t I = 0;
    for (auto &class_def: classes.get_classdefs()) {
        fmt::print("Class #{} data:\n", I++);

        auto *const class_idx = class_def.get_class_idx();
        auto *const super_class = class_def.get_superclass();
        std::string_view source_file = class_def.get_source_file();
        auto access_flags = class_def.get_access_flags();

        fmt::print(" Class name:            {}\n", class_idx->get_class_name());
        fmt::print(" Super class:           {}\n", super_class->get_class_name());
        if (!source_file.empty())
            fmt::print(" Source file:           {}\n", source_file);
        fmt::print(" Access flags:          0x{:X} ({})\n", static_cast<std::uint32_t>(access_flags),
                   shuriken::dex::Utils::get_types_as_string(access_flags));

        auto &class_def_struct = class_def.get_class_def_struct();
        fmt::print(" Super class idx:       {}\n", class_def_struct.superclass_idx);
        fmt::print(" Interfacess off:       0x{:X}\n", class_def_struct.interfaces_off);
        fmt::print(" Annotations off:       0x{:X}\n", class_def_struct.annotations_off);
        fmt::print(" Class data off:        0x{:X}\n", class_def_struct.class_data_off);
        fmt::print(" Static values off:     0x{:X}\n", class_def_struct.static_values_off);

        auto &class_data_item = class_def.get_class_data_item();

        fmt::print(" Static fields size:    {}\n", class_data_item.get_number_of_static_fields());
        fmt::print(" Instance fields size:  {}\n", class_data_item.get_number_of_instance_fields());
        fmt::print(" Direct methods size:   {}\n", class_data_item.get_number_of_direct_methods());
        fmt::print(" Virtual methods size:  {}\n", class_data_item.get_number_of_virtual_methods());


        if (fields) {
            fmt::print(" Static Fields:\n");
            for (size_t j = 0, e = class_data_item.get_number_of_static_fields(); j < e; j++) {
                auto *field = class_data_item.get_static_field_by_id(static_cast<uint32_t>(j));
                print_field(field, j);
            }

            fmt::print(" Instance Fields:\n");
            for (size_t j = 0, e = class_data_item.get_number_of_instance_fields(); j < e; j++) {
                auto *field = class_data_item.get_instance_field_by_id(static_cast<uint32_t>(j));
                print_field(field, j);
            }
        }

        if (methods) {
            fmt::print(" Direct Methods:\n");
            for (size_t j = 0, e = class_data_item.get_number_of_direct_methods(); j < e; j++) {
                auto *method = class_data_item.get_direct_method_by_id(static_cast<uint32_t>(j));
                print_method(method, j);
            }

            fmt::print(" Virtual Methods:\n");
            for (size_t j = 0, e = class_data_item.get_number_of_virtual_methods(); j < e; j++) {
                auto *method = class_data_item.get_virtual_method_by_id(static_cast<uint32_t>(j));
                print_method(method, j);
            }
        }

        if (xrefs) {
            fmt::print(" XREFs\n");
            auto *class_analysis = analysis->get_class_analysis(class_idx->get_class_name().data());

            auto xrefconstclass = class_analysis->get_xrefconstclass();
            fmt::print("  XREF Const Class:\n");
            for (auto & xref : xrefconstclass) {
                fmt::print("   - {}:{}\n", xref.first->get_full_name(), xref.second);
            }
            auto xrefnewinstance = class_analysis->get_xrefnewinstance();
            fmt::print("  XREF New Instance:\n");
            for (auto & xref : xrefnewinstance) {
                fmt::print("   - {}:{}\n", xref.first->get_full_name(), xref.second);
            }
        }

    }
}

void print_field(shuriken::parser::dex::EncodedField *field, size_t j) {
    fmt::print("  Field #{}\n", j);
    fmt::print("   Name:            {}\n", field->get_field()->field_name());
    fmt::print("   Type:            {}\n", field->get_field()->field_type()->get_raw_type());
    fmt::print("   Access Flags:    {} ({})\n", static_cast<std::uint32_t>(field->get_flags()),
               shuriken::dex::Utils::get_types_as_string(field->get_flags()));
    if (xrefs) {
        fmt::print("   XRefs:\n");
        auto *field_analysis = analysis->get_field_analysis(field);
        if (field_analysis == nullptr) return;
        auto xref_read = field_analysis->get_xrefread();
        fmt::print("    Xrefs Read:\n");
        for (auto & xref : xref_read) {
            fmt::print("      {}:{}\n",
                       std::get<shuriken::analysis::dex::MethodAnalysis*>(xref)->get_full_name(),
                       std::get<std::uint64_t>(xref));
        }
        auto xref_write = field_analysis->get_xrefwrite();
        fmt::print("    Xrefs Write:\n");
        for (auto & xref : xref_write) {
            fmt::print("      {}:{}\n",
                       std::get<shuriken::analysis::dex::MethodAnalysis*>(xref)->get_full_name(),
                       std::get<std::uint64_t>(xref));
        }
    }
}

void print_method(shuriken::parser::dex::EncodedMethod *method, size_t j) {
    fmt::print("  Method #{}\n", j);
    auto *method_id = method->getMethodID();
    fmt::print("   Method name:    {}\n", method_id->get_method_name());
    fmt::print("   Prototype:      (");
    for (auto *p: method_id->get_prototype()->get_parameters()) {
        fmt::print("{}", p->get_raw_type());
    }
    fmt::print("){}\n", method_id->get_prototype()->get_return_type()->get_raw_type());
    fmt::print("   Access Flags:   0x{:X} ({})\n", static_cast<std::uint32_t>(method->get_flags()),
               shuriken::dex::Utils::get_types_as_string(method->get_flags()));
    auto *code_item_struct = method->get_code_item();
    if (code_item_struct) {
        fmt::print("   Code:           {}\n", "-");
        fmt::print("   Registers:      {}\n", code_item_struct->get_registers_size());
        fmt::print("   Ins:            {}\n", code_item_struct->get_incomings_args());
        fmt::print("   Outs:           {}\n", code_item_struct->get_outgoing_args());
        fmt::print("   Code size:      {}\n", code_item_struct->get_instructions_size());
    } else {
        fmt::print("   Code:           {}\n", "<None>");
    }
    if (code) {
        print_code(code_item_struct->get_bytecode());
    }
    if (disassembly) {
        fmt::println("Disassembled method:");
        auto *disassembled_method = disassembler->get_disassembled_method(method_id->dalvik_name_format());
        if (disassembled_method == nullptr)
            throw std::runtime_error("The method " + std::string(method_id->demangle()) + " was not correctly disassembled");
        fmt::print("{}\n", disassembled_method->print_method());
    }
    if (blocks) {
        auto *method_analysis = analysis->get_method(method);
        if (method_analysis == nullptr) return;

        if (method_analysis) {
            fmt::print("\n{}\n", method_analysis->toString());
        }
    }
    if (xrefs) {
        auto *method_analysis = analysis->get_method(method);
        if (method_analysis == nullptr) return;

        fmt::print("    XREFs\n");

        auto xrefto = method_analysis->get_xrefto();
        fmt::print("     XREF To:\n");
        for (auto & xref : xrefto) {
                fmt::print("      - {}:{}\n",
                           std::get<shuriken::analysis::dex::MethodAnalysis*>(xref)->get_full_name(),
                           std::get<std::uint64_t>(xref));

        }
        auto xreffrom = method_analysis->get_xreffrom();
        fmt::print("     XREF From:\n");
        for (auto & xref : xreffrom) {
            fmt::print("      - {}:{}\n",
                       std::get<shuriken::analysis::dex::MethodAnalysis*>(xref)->get_full_name(),
                       std::get<std::uint64_t>(xref));

        }
    }
}

void print_code(std::span<std::uint8_t> bytecode) {
    fmt::print("   Code: ");
    size_t j = 0;
    for (auto b: bytecode) {
        fmt::print("{:02X} ", b);
        if (j++ == 8) {
            j = 0;
            fmt::print("\n         ");
        }
    }
    fmt::print("\n");
}

#ifdef MACHO
    void parse_macho(std::string& macho_file) {
        parsed_macho = shuriken::parser::parse_macho(macho_file);
        const shuriken::parser::macho::MachoHeader &header = parsed_macho->get_header();
        const shuriken::parser::macho::MachoCommands &commands = parsed_macho->get_commands();
        const shuriken::parser::macho::MachoSections &sections = parsed_macho->get_sections();

        if (!no_print) {
            if (macho_headers) print_macho_header(header);
            if (macho_loadcommands) print_macho_loadcommands(commands);
            if (macho_segmentcommands) print_macho_segmentcommands(commands);
            if (macho_sections) print_macho_sections(sections);
        }
    }

    void print_macho_header(const shuriken::parser::macho::MachoHeader &header) {
        auto &macho_header = header.get_macho_header_const();

        fmt::println("Mach-O Header:\n");
        fmt::print(" Magic:                 0x{:x}\n", macho_header.magic);
        fmt::print(" CPU type:              0x{:X}\n", macho_header.cputype);
        fmt::print(" CPU subtype:           0x{:X}\n", macho_header.cpusubtype);
        fmt::print(" Filetype:              0x{:X}\n", macho_header.filetype);
        fmt::print(" Number of commands:    {}\n", macho_header.ncmds);
        fmt::print(" Size of commands:      {} bytes\n", macho_header.sizeofcmds);
        fmt::print(" Flags:                 0x{:X}\n", macho_header.flags);
        fmt::print(" Reserved:              0x{:X}\n", macho_header.reserved);
    }

    void print_macho_loadcommands(const shuriken::parser::macho::MachoCommands &commands) {
        const shuriken::parser::macho::MachoCommands::loadcommands_t& macho_loadcommands = commands.get_macho_loadcommands_const();

        for (const std::unique_ptr<shuriken::parser::macho::MachoCommands::loadcommand_t>& loadcommand : macho_loadcommands) {
            fmt::println("Load command:\n");
            fmt::print(" Type:                  0x{:X}\n", loadcommand->cmd);
            fmt::print(" Size:                  {} bytes\n", loadcommand->cmdsize);
        }
    }

    void print_macho_segmentcommands(const shuriken::parser::macho::MachoCommands &commands) {
        const shuriken::parser::macho::MachoCommands::segmentcommands_t& macho_segmentcommands = commands.get_macho_segmentcommands_const();

        for (const std::unique_ptr<shuriken::parser::macho::MachoCommands::segmentcommand_t>& segmentcommand : macho_segmentcommands) {
            fmt::println("Segment command:\n");
            fmt::print(" Type:                  0x{:X}\n", segmentcommand->cmd);
            fmt::print(" Size:                  {} bytes\n", segmentcommand->cmdsize);
            fmt::print(" Name:                  {}\n", segmentcommand->segname);
            fmt::print(" VM address:            0x{:X}\n", segmentcommand->vmaddr);
            fmt::print(" VM size:               0x{:X}\n", segmentcommand->vmsize);
            fmt::print(" Offset in file:        0x{:X}\n", segmentcommand->fileoff);
            fmt::print(" Size in file:          0x{:X}\n", segmentcommand->filesize);
            fmt::print(" Max memory protection: 0x{:X}\n", segmentcommand->maxprot);
            fmt::print(" Initial memory prot:   0x{:X}\n", segmentcommand->initprot);
            fmt::print(" Number of Sections:    0x{:X}\n", segmentcommand->nsects);
            fmt::print(" Flags:                 0x{:X}\n", segmentcommand->flags);
        }
    }

    void print_macho_sections(const shuriken::parser::macho::MachoSections &sections) {
        const shuriken::parser::macho::MachoSections::sections_t&  macho_sections = sections.get_sections_const();

        for (const std::shared_ptr<shuriken::parser::macho::MachoSections::section_t>& section: macho_sections) {
            fmt::println("Section:\n");
            fmt::print(" Type:                  {}\n", section->sectname);
            fmt::print(" Seg:                   {}\n", section->segname);
            fmt::print(" Address:               0x{:X}\n", section->addr);
            fmt::print(" Size:                  {} bytes\n", section->size);
            fmt::print(" File offset:           0x{:X}\n", section->offset);
            fmt::print(" Alignment:             {} bytes\n", (1 << section->align));
            fmt::print(" Relocation offset:     0x{:X}\n", section->reloff);
            fmt::print(" Number of relocations: {}\n", section->nreloc);
            fmt::print(" Flags:                 0x{:X}\n", section->flags);
            fmt::print(" Reserved1:             0x{:X}\n", section->reserved1);
            fmt::print(" Reserved2:             0x{:X}\n", section->reserved2);
            fmt::print(" Reserved3:             0x{:X}\n", section->reserved3);
        }
    }
#endif