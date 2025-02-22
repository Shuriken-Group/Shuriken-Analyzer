#include "plugin.h"
#include "shurikenservice.h"
#include <span>

#include <fstream>
#include <iomanip>

std::mutex decodeMutex;

// Hack for proof of concept ----------------------------------------------
BinaryNinja::BinaryView *g_bv = nullptr;

extern "C" {
__declspec(dllexport) void SetBinaryView(BinaryNinja::BinaryView *bv) {
    g_bv = bv;
}
}
// -------------------------------------------------------------------------

DalvikArchitecture::DalvikArchitecture(const char *name, BNEndianness endian) : BinaryNinja::Architecture(name) {
    m_endian = endian;
    BinaryNinja::LogInfo("%p - *********************** DalvikArchitecture::DalvikArchitecture", this);
}

BNEndianness DalvikArchitecture::GetEndianness() const {
    return m_endian;
}

size_t DalvikArchitecture::GetAddressSize() const {
    return 4;
}

size_t DalvikArchitecture::GetMaxInstructionLength() const {
    return 10;
}

size_t DalvikArchitecture::GetDefaultIntegerSize() const {
    return 1;
}

size_t DalvikArchitecture::GetInstructionAlignment() const {
    return 1;
}

// TODO: ugly hack fix
std::unique_ptr<BinaryNinja::BinaryReader> g_reader = nullptr;

bool DalvikArchitecture::GetInstructionInfo(const uint8_t *data, uint64_t addr, size_t maxLen, BinaryNinja::InstructionInfo &result) {

    try {
        // TODO: ugly hack fix
        IShurikenService &shurikenService = GetShurikenService();
        const IShurikenView &shurikenView = shurikenService.getView(g_bv);
        if (g_reader == nullptr) {
            g_reader = std::make_unique<BinaryNinja::BinaryReader>(g_bv, LittleEndian);
        }

        // TODO: fix const usage of disassembler data
        std::vector<uint8_t> buffer(maxLen);
        std::copy(data, data + maxLen, buffer.begin());
        std::span<uint8_t> byteCode(buffer.data(), buffer.size());

        std::lock_guard<std::mutex> lock(decodeMutex);

        {
            std::ofstream logFile("C:\\Users\\main\\store\\code\\github\\Shuriken-Analyzer\\build_Release\\binjaplugin-build\\instruction_log.txt", std::ios_base::app);
            if (logFile.is_open()) {
                logFile << addr << " ";
                logFile << std::endl;
                logFile.close();
            }
        }

        auto insn = shurikenView.getParser().getDisassembler().decodeInstruction(byteCode);

        if (!insn) {
            result.length = 0;
            return false;
        }

        // TODO: completely refactor and clean up this whole function in general
        switch (insn->getMnemonic().value()) {
            case shurikenapi::disassembly::Mnemonic::OP_RETURN_VOID:
            case shurikenapi::disassembly::Mnemonic::OP_RETURN:
            case shurikenapi::disassembly::Mnemonic::OP_RETURN_WIDE:
            case shurikenapi::disassembly::Mnemonic::OP_RETURN_OBJECT:
                result.AddBranch(FunctionReturn);
                break;
            case shurikenapi::disassembly::Mnemonic::OP_THROW:
                result.AddBranch(ExceptionBranch);// TODO: exception branch
                break;
            case shurikenapi::disassembly::Mnemonic::OP_GOTO_16:
            case shurikenapi::disassembly::Mnemonic::OP_GOTO: {
                const auto &operands = insn->getOperands();
                // sanity check - shouldnt happen
                if (operands.size() != 1) {
                    BinaryNinja::LogError("OP_GOTO_X: operands.size() != 1 - processing addr: %llx", addr);
                    return false;
                }
                const shurikenapi::disassembly::UBranchOperand *ubranchOp =
                        dynamic_cast<const shurikenapi::disassembly::UBranchOperand *>(&insn->getOperands()[0].get());
                if (!ubranchOp) {
                    BinaryNinja::LogError("OP_GOTO_X: cast error - processing addr: %llx", addr);
                    return false;
                }

                result.AddBranch(UnconditionalBranch, ubranchOp->calculateTarget(addr));
                break;
            }
            case shurikenapi::disassembly::Mnemonic::OP_IF_EQ:
            case shurikenapi::disassembly::Mnemonic::OP_IF_NE:
            case shurikenapi::disassembly::Mnemonic::OP_IF_LT:
            case shurikenapi::disassembly::Mnemonic::OP_IF_GT:
            case shurikenapi::disassembly::Mnemonic::OP_IF_LE:
            case shurikenapi::disassembly::Mnemonic::OP_IF_GE: {
                const auto &operands = insn->getOperands();
                // sanity check - shouldnt happen
                if (operands.size() != 3) {
                    BinaryNinja::LogError("OP_IF_LEZ: operands.size() != 1 - processing addr: %llx", addr);
                    return false;
                }
                const shurikenapi::disassembly::CBranchOperand *cbranchOp =
                        dynamic_cast<const shurikenapi::disassembly::CBranchOperand *>(&insn->getOperands()[2].get());
                if (!cbranchOp) {
                    BinaryNinja::LogError("OP_IF_LEZ: cast error - processing addr: %llx", addr);
                    return false;
                }
                result.AddBranch(TrueBranch, cbranchOp->calculateTrueTarget(addr));
                result.AddBranch(FalseBranch, addr + insn->getSize());
                break;
            }
            case shurikenapi::disassembly::Mnemonic::OP_IF_EQZ:
            case shurikenapi::disassembly::Mnemonic::OP_IF_NEZ:
            case shurikenapi::disassembly::Mnemonic::OP_IF_LTZ:
            case shurikenapi::disassembly::Mnemonic::OP_IF_GEZ:
            case shurikenapi::disassembly::Mnemonic::OP_IF_GTZ:
            case shurikenapi::disassembly::Mnemonic::OP_IF_LEZ: {
                const auto &operands = insn->getOperands();
                // sanity check - shouldnt happen
                if (operands.size() != 1) {
                    BinaryNinja::LogError("OP_IF_LEZ: operands.size() != 1 - processing addr: %llx", addr);
                    return false;
                }
                const shurikenapi::disassembly::CBranchOperand *cbranchOp =
                        dynamic_cast<const shurikenapi::disassembly::CBranchOperand *>(&insn->getOperands()[0].get());
                if (!cbranchOp) {
                    BinaryNinja::LogError("OP_IF_LEZ: cast error - processing addr: %llx", addr);
                    return false;
                }
                result.AddBranch(TrueBranch, cbranchOp->calculateTrueTarget(addr));
                result.AddBranch(FalseBranch, addr + insn->getSize());
                break;
            }
            default:
                break;
        }

        result.length = insn->getSize();

    } catch (const std::exception &e) {
        BinaryNinja::LogInfo("Exception GetInstructionInfo: %s processing offset: %llx", e.what(), addr);
        return false;
    }

    return true;
}

bool DalvikArchitecture::GetInstructionText(const uint8_t *data, uint64_t addr, size_t &len,
                                            std::vector<BinaryNinja::InstructionTextToken> &result) {

    // TODO: fix const usage of disassembler data
    std::vector<uint8_t> buffer(len);
    std::copy(data, data + len, buffer.begin());
    std::span<uint8_t> byteCode(buffer.data(), buffer.size());

    try {
        IShurikenService &shurikenService = GetShurikenService();
        const IShurikenView &shurikenView = shurikenService.getView(g_bv);

        std::lock_guard<std::mutex> lock(decodeMutex);
        auto insn = shurikenView.getParser().getDisassembler().decodeInstruction(byteCode);
        if (!insn) {
            return false;
        }

        char buf[31];
        memset(buf, 0x20, sizeof(buf));
        size_t operationLen = insn->getMnemonic().string().length();
        if (operationLen < 30) {
            buf[30 - operationLen] = '\0';
        } else
            buf[1] = '\0';
        result.emplace_back(InstructionToken, insn->getMnemonic().string());
        result.emplace_back(TextToken, buf);

        const auto &operands = insn->getOperands();
        for (size_t i = 0; i < operands.size(); ++i) {
            const auto &op = operands[i];

            const shurikenapi::disassembly::RegisterOperand<std::uint8_t> *regOp =
                    dynamic_cast<const shurikenapi::disassembly::RegisterOperand<std::uint8_t> *>(&op.get());
            if (regOp) {
                result.emplace_back(RegisterToken, regOp->string());
            }

            const shurikenapi::disassembly::RegisterOperand<std::uint16_t> *regOp16 =
                    dynamic_cast<const shurikenapi::disassembly::RegisterOperand<std::uint16_t> *>(&op.get());
            if (regOp16) {
                result.emplace_back(RegisterToken, regOp16->string());
            }

            const shurikenapi::disassembly::RegisterListOperand<std::uint8_t> *regListOp =
                    dynamic_cast<const shurikenapi::disassembly::RegisterListOperand<std::uint8_t> *>(&op.get());
            if (regListOp) {
                result.emplace_back(TextToken, "{");
                for (size_t i = 0; i < regListOp->getRegs().size(); ++i) {
                    result.emplace_back(RegisterToken, "v" + std::to_string(regListOp->getRegs()[i]));
                    if (i != regListOp->getRegs().size() - 1) {
                        result.emplace_back(OperandSeparatorToken, ", ");
                    }
                }
                result.emplace_back(TextToken, "}");
            }
            const shurikenapi::disassembly::RegisterListOperand<std::uint16_t> *regList16Op =
                    dynamic_cast<const shurikenapi::disassembly::RegisterListOperand<std::uint16_t> *>(&op.get());
            if (regList16Op) {
                result.emplace_back(TextToken, "{");
                for (size_t i = 0; i < regList16Op->getRegs().size(); ++i) {
                    result.emplace_back(RegisterToken, "v" + std::to_string(regList16Op->getRegs()[i]));
                    if (i != regList16Op->getRegs().size() - 1) {
                        result.emplace_back(OperandSeparatorToken, ", ");
                    }
                }
                result.emplace_back(TextToken, "}");
            }

            const shurikenapi::disassembly::Imm8Operand *imm8Op =
                    dynamic_cast<const shurikenapi::disassembly::Imm8Operand *>(&op.get());
            if (imm8Op) {
                result.emplace_back(IntegerToken, imm8Op->string(), static_cast<std::int8_t>(imm8Op->value(), 1));
            }

            const shurikenapi::disassembly::Imm16Operand *imm16Op =
                    dynamic_cast<const shurikenapi::disassembly::Imm16Operand *>(&op.get());
            if (imm16Op) {
                result.emplace_back(IntegerToken, imm16Op->string(), static_cast<std::int16_t>(imm16Op->value(), 2));
            }

            const shurikenapi::disassembly::Imm32Operand *imm32Op =
                    dynamic_cast<const shurikenapi::disassembly::Imm32Operand *>(&op.get());
            if (imm32Op) {
                result.emplace_back(IntegerToken, imm32Op->string(), static_cast<std::int16_t>(imm32Op->value(), 4));
            }

            const shurikenapi::disassembly::Imm64Operand *imm64Op =
                    dynamic_cast<const shurikenapi::disassembly::Imm64Operand *>(&op.get());
            if (imm64Op) {
                result.emplace_back(IntegerToken, imm64Op->string(), imm64Op->value(), 8);
            }

            const shurikenapi::disassembly::DVMTypeOperand *dvmTypeOp =
                    dynamic_cast<const shurikenapi::disassembly::DVMTypeOperand *>(&op.get());
            if (dvmTypeOp) {
                result.emplace_back(StructureHexDumpTextToken, dvmTypeOp->string());
            }

            const shurikenapi::disassembly::FieldOperand *fieldOp =
                    dynamic_cast<const shurikenapi::disassembly::FieldOperand *>(&op.get());
            if (fieldOp) {
                result.emplace_back(DataSymbolToken, fieldOp->string());
            }

            const shurikenapi::disassembly::UBranchOperand *ubranchOp =
                    dynamic_cast<const shurikenapi::disassembly::UBranchOperand *>(&op.get());
            if (ubranchOp) {
                std::stringstream ss;
                ss << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << ubranchOp->calculateTarget(addr);

                result.emplace_back(PossibleAddressToken, ss.str(), ubranchOp->calculateTarget(addr));
            }

            const shurikenapi::disassembly::CBranchOperand *cbranchOp =
                    dynamic_cast<const shurikenapi::disassembly::CBranchOperand *>(&op.get());
            if (cbranchOp) {
                std::stringstream ss;
                ss << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << cbranchOp->calculateTrueTarget(addr);

                result.emplace_back(PossibleAddressToken, ss.str(), cbranchOp->calculateTrueTarget(addr));
            }

            const shurikenapi::disassembly::MethodOperand *methodOp =
                    dynamic_cast<const shurikenapi::disassembly::MethodOperand *>(&op.get());
            if (methodOp) {
                auto sym = g_bv->GetSymbolByRawName(methodOp->string());

                if (sym) {
                    if (sym->GetNameSpace() == g_bv->GetExternalNameSpace())
                        result.emplace_back(ExternalSymbolToken, methodOp->string(), sym->GetAddress());
                    else
                        result.emplace_back(CodeSymbolToken, methodOp->string(), sym->GetAddress());
                } else {
                    result.emplace_back(CodeSymbolToken, methodOp->string(), 0);
                    BinaryNinja::LogWarn("Symbol not found: %s", methodOp->string().c_str());
                }
            }

            const shurikenapi::disassembly::StringOperand *stringOp =
                    dynamic_cast<const shurikenapi::disassembly::StringOperand *>(&op.get());
            if (stringOp) {
                uint32_t offset = shurikenView.getStringOffset(stringOp->value());
                BinaryNinja::InstructionTextToken token;
                token.value = offset;
                token.address = offset;
                token.text = stringOp->string();
                token.type = StringToken;
                result.push_back(token);
            }

            const shurikenapi::disassembly::SwitchOperand *switchOp =
                    dynamic_cast<const shurikenapi::disassembly::SwitchOperand *>(&op.get());
            if (switchOp) {
                std::stringstream ss;
                ss << "switchtable_0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0')
                   << (switchOp->value() * 2) + addr;
                result.emplace_back(PossibleAddressToken, ss.str(), (switchOp->value() * 2) + addr);
            }

            if (i != operands.size() - 1) {
                result.emplace_back(OperandSeparatorToken, ", ");
            }
        }

        len = insn->getSize();
    } catch (const std::exception &e) {
        BinaryNinja::LogInfo("Exception GetInstructionText: %s processing offset: %llx", e.what(), addr);
        return false;
    }

    return true;
}
