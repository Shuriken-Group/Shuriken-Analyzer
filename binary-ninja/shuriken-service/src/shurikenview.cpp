#include "render.h"
#include "shurikenservice_internal.h"

using BinaryNinja::Function;
using BinaryNinja::FunctionParameter;
using BinaryNinja::NameSpace;
using BinaryNinja::QualifiedName;
using BinaryNinja::Ref;
using BinaryNinja::Structure;
using BinaryNinja::StructureBuilder;
using BinaryNinja::Symbol;
using BinaryNinja::Type;

ShurikenView::ShurikenView(BinaryNinja::BinaryView *bv) : m_bv(bv) {

    m_parser = shurikenapi::parse_dex(bv->GetFile()->GetOriginalFilename());
}

const shurikenapi::IDex &ShurikenView::getParser() const {
    return *m_parser;
}

void ShurikenView::buildMetaData() {
    auto reader = BinaryReader(m_bv);

    // --Create Fundamental Types------
    m_bv->DefineType("boolean", QualifiedName("boolean"), Type::BoolType());
    m_bv->DefineType("byte", QualifiedName("byte"), Type::IntegerType(1, false));
    m_bv->DefineType("char", QualifiedName("char"), Type::IntegerType(1, true));
    m_bv->DefineType("double", QualifiedName("double"), Type::FloatType(8));
    m_bv->DefineType("float", QualifiedName("float"), Type::FloatType(4));
    m_bv->DefineType("int", QualifiedName("int"), Type::IntegerType(4, true));
    m_bv->DefineType("long", QualifiedName("long"), Type::IntegerType(8, true));
    m_bv->DefineType("short", QualifiedName("short"), Type::IntegerType(2, true));
    m_bv->DefineType("void", QualifiedName("void"), Type::VoidType());

    // --Create DEX Header Structure------
    StructureBuilder dexHeaderBuilder;
    dexHeaderBuilder.AddMember(Type::ArrayType(Type::IntegerType(1, true), 8), "magic");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "checksum");
    dexHeaderBuilder.AddMember(Type::ArrayType(Type::IntegerType(1, true), 20), "signature");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "file_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "header_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "endian_tag");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "link_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "link_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "map_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "string_ids_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "string_ids_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "type_ids_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "type_ids_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "proto_ids_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "proto_ids_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "field_ids_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "field_ids_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "method_ids_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "method_ids_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "class_defs_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "class_defs_off");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "data_size");
    dexHeaderBuilder.AddMember(Type::IntegerType(4, false), "data_off");
    Ref<Structure> dexHeaderStruct = dexHeaderBuilder.Finalize();
    Ref<Type> dexHeaderType = Type::StructureType(dexHeaderStruct);
    QualifiedName dexHeaderName = std::string("DEX_Header");
    QualifiedName dexHeaderTypeName = m_bv->DefineType(Type::GenerateAutoTypeId("dex", dexHeaderName), dexHeaderName, dexHeaderType);
    m_bv->DefineDataVariable(0, Type::NamedType(m_bv, dexHeaderTypeName));
    m_bv->DefineAutoSymbol(new Symbol(DataSymbol, "__dex_header", 0, NoBinding));

    // --Create StringTable Array------
    // Read data
    reader.Seek(0x38);
    uint32_t stringTableSize = reader.Read32();
    reader.Seek(0x3C);
    uint32_t stringTableOffset = reader.Read32();
    // Create Array Type
    Ref<Type> stringTableType = Type::ArrayType(Type::IntegerType(4, true), stringTableSize);
    stringTableType = Type::NamedType(m_bv, m_bv->DefineType("StringTable", QualifiedName("StringTable"), stringTableType));
    m_bv->DefineDataVariable(stringTableOffset, stringTableType);
    // Set Variable Name
    Ref<Symbol> variableSymbol = new Symbol(BNSymbolType::DataSymbol, "stringTable", stringTableOffset, NoBinding);
    m_bv->DefineAutoSymbol(variableSymbol);
    // Set Renderer
    StringTable *stringTable = new StringTable();
    BinaryNinja::DataRendererContainer::RegisterTypeSpecificDataRenderer(stringTable);

    // --Define Strings in view------
    // Read Data
    std::vector<uint32_t> stringOffsets;
    reader.Seek(0x38);
    int amount = reader.Read32();
    reader.Seek(0x3C);
    int offset = reader.Read32();
    reader.Seek(offset);
    for (auto i = 0; i < amount; i++) {
        stringOffsets.push_back(reader.Read32());
        offset += 4;
        reader.Seek(offset);
    }

    // Define Strings
    for (int i = 0; i < amount; i++) {
        int entry = stringOffsets[i];
        m_bv->DefineDataVariable(entry, Type::IntegerType(1, false));
        reader.Seek(entry);
        int length = reader.Read8();
        m_bv->DefineDataVariable(entry + 1, Type::ArrayType(Type::IntegerType(1, true), length + 1));

        m_stringMap.insert({i, entry + 1});
    }
}

void ShurikenView::buildSymbols() {

    auto classes = m_parser->getClassManager().getAllClasses();

    // Add External Symbols
    // uint64_t offset = m_bv->GetEnd() + 0x1000;
    uint64_t offset = 0;
    for (const auto &c: classes) {
        if (!c.get().isExternal())
            continue;

        for (auto &m: c.get().getExternalMethods()) {
            NameSpace ns = m_bv->GetExternalNameSpace();
            Ref<Symbol> sym = new Symbol(ExternalSymbol, m.get().getDalvikName(), m.get().getDalvikName(), m.get().getDalvikName(),
                                         offset, NoBinding, ns, 0);
            Ref<Type> proto = Type::FunctionType(Type::VoidType(), m_bv->GetDefaultPlatform()->GetDefaultCallingConvention(), {});
            auto out = m_bv->DefineAutoSymbolAndVariableOrFunction(m_bv->GetDefaultPlatform(), sym, proto);
            // offset += 8;
        }
    }

    // Add Internal Symbols
    for (const auto &c: classes) {

        if (c.get().isExternal())
            continue;

        BinaryNinja::LogInfo("Class: %s", c.get().getName().c_str());
        auto component = m_bv->CreateComponentWithName(c.get().getName());
        for (auto &m: c.get().getDirectMethods()) {
            Ref<Function> func = buildMethod(m.get());
            if (func) {
                component->AddFunction(func);
            }
        }
    }
}

Ref<Type> ShurikenView::getFundamental(const shurikenapi::FundamentalValue &value) {
    switch (value) {
        case shurikenapi::FundamentalValue::kBoolean:
            return Type::BoolType();
        case shurikenapi::FundamentalValue::kByte:
            return Type::IntegerType(1, false);
        case shurikenapi::FundamentalValue::kChar:
            return Type::IntegerType(1, true);
        case shurikenapi::FundamentalValue::kDouble:
            return Type::FloatType(8);
        case shurikenapi::FundamentalValue::kFloat:
            return Type::FloatType(4);
        case shurikenapi::FundamentalValue::kInt:
            return Type::IntegerType(4, true);
        case shurikenapi::FundamentalValue::kLong:
            return Type::IntegerType(8, true);
        case shurikenapi::FundamentalValue::kShort:
            return Type::IntegerType(2, true);
        case shurikenapi::FundamentalValue::kVoid:
            return Type::VoidType();
        default:
            BinaryNinja::LogWarn("Unknown fundamental value: %d", value);
            return Type::VoidType();
    }
}

Ref<Function> ShurikenView::buildMethod(const shurikenapi::IClassMethod &method) {

    uint64_t funcOffset = method.getCodeLocation();
    BinaryNinja::LogInfo("---------------");
    BinaryNinja::LogInfo("Building method: %s at %016llx", method.getDalvikName().c_str(), funcOffset);

    // Get method prototype
    auto &prototype = method.getPrototype();
    auto &returnType = prototype.getReturnType();
    if (returnType.getType() != shurikenapi::DexType::kFundamental) {
        // TODO: implement type system
        // BinaryNinja::LogError("Unsupported return type for method: %s", method.getDalvikName().c_str());
        // return Ref<Function>();
    }

    // Create function object with calling convention
    Ref<Function> func = m_bv->CreateUserFunction(m_bv->GetDefaultPlatform(), funcOffset);
    func->SetCallingConvention(m_bv->GetDefaultPlatform()->GetDefaultCallingConvention());

    // Create parameters
    std::vector<FunctionParameter> parameters;
    for (const auto &p: prototype.getParameters()) {
        // TODO: implement type system
        /*
        if (p.get().getType() != shurikenapi::DexType::kFundamental) {
            BinaryNinja::LogError("Unsupported parameter type for method: %s", method.getDalvikName().c_str());
            return Ref<Function>();
        }
        Ref<Type> paramType = getFundamental(p.get().getFundamentalValue().value());
        BinaryNinja::LogInfo("ParameterType: %d", p.get().getFundamentalValue().value());
        */
        Ref<Type> paramType = Type::IntegerType(4, true);
        parameters.push_back(FunctionParameter("", paramType));
    }

    // Set function type
    // BinaryNinja::LogInfo("ReturnType: %d", returnType.getFundamentalValue().value());
    Ref<Type> funcType =
            Type::FunctionType(Type::IntegerType(4, true), m_bv->GetDefaultPlatform()->GetDefaultCallingConvention(), parameters);
    func->SetUserType(funcType);

    // Define the function
    m_bv->DefineUserSymbol(new Symbol(FunctionSymbol, method.getDalvikName(), funcOffset, NoBinding));

    BinaryNinja::LogInfo("Building method: %s - OK", method.getDalvikName().c_str());

    return func;
}


uint32_t ShurikenView::getStringOffset(uint32_t id) const {

    auto it = m_stringMap.find(id);
    if (it == m_stringMap.end()) {
        return 0;
    }

    return it->second;
}
