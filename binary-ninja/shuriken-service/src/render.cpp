#include "render.h"

#include <iomanip>
#include <sstream>
#include <string>

bool StringTable::IsValidForData(BinaryView *data, uint64_t addr, Type *type, std::vector<std::pair<Type *, size_t>> &context) {

    BinaryNinja::Ref<Type> retrievedType = data->GetTypeByName(BinaryNinja::QualifiedName("StringTable"));

    if (!retrievedType || !type)
        return false;

    if (*type == *retrievedType)
        return true;

    return false;
}

// ugly render code :')
// TODO: refactor
std::vector<DisassemblyTextLine> StringTable::GetLinesForData(BinaryView *data, uint64_t addr, Type *type,
                                                              const std::vector<InstructionTextToken> &prefix, size_t width,
                                                              std::vector<std::pair<Type *, size_t>> &context, const std::string &language) {

    std::vector<DisassemblyTextLine> result;

    uint64_t offset = addr;
    int amount = 0;
    std::vector<uint32_t> stringOffsets;

    auto reader = BinaryReader(data);
    reader.Seek(0x38);
    amount = reader.Read32();
    reader.Seek(offset);
    for (auto i = 0; i < amount; i++) {
        stringOffsets.push_back(reader.Read32());
        offset += 4;
        reader.Seek(offset);
    }

    DisassemblyTextLine line;
    line.addr = addr;
    line.tokens.emplace_back(TextToken, "");
    result.push_back(line);
    line.tokens.emplace_back(TextToken, "┌─[StringTable]────┬──────────────────┬──────────────────┬──────────────────┐");
    result.push_back(line);

    std::stringstream ss;
    int idx = 0;
    if (stringOffsets.size() < 4) {

        DisassemblyTextLine line;
        line.addr = addr + (idx * 4);
        int remainingCount = 4 - static_cast<int>(stringOffsets.size() - idx);
        for (; idx < stringOffsets.size(); ++idx) {
            ss.str("");
            ss.clear();
            ss << std::setw(4) << std::setfill('0') << idx;
            line.tokens.emplace_back(TextToken, "│ ID " + ss.str() + ": ");
            ss.str("");
            ss.clear();
            ss << "0x" << std::hex << std::uppercase << stringOffsets[idx];
            line.tokens.emplace_back(PossibleAddressToken, ss.str() + " ", stringOffsets[idx]);
        }
        line.tokens.emplace_back(TextToken, "│");
        for (int x = 0; x < remainingCount; x++) {
            ss.str("");
            ss.clear();
            line.tokens.emplace_back(TextToken, "                │");
        }
        result.push_back(line);

        DisassemblyTextLine line2;
        line2.addr = addr + (idx * 4);
        line2.tokens.emplace_back(TextToken, "└──────────────────┴──────────────────┴──────────────────┴──────────────────┘");
        result.push_back(line2);

    } else {
        for (; idx + 3 < stringOffsets.size(); idx += 4) {
            uint32_t a = stringOffsets[idx];
            uint32_t b = stringOffsets[idx + 1];
            uint32_t c = stringOffsets[idx + 2];
            uint32_t d = stringOffsets[idx + 3];

            DisassemblyTextLine line;
            line.addr = addr + (idx * 4);
            ss.str("");
            ss.clear();
            ss << std::setw(4) << std::setfill('0') << idx;
            line.tokens.emplace_back(TextToken, "│ ID " + ss.str() + ": ");
            ss.str("");
            ss.clear();
            ss << "0x" << std::hex << std::uppercase << a;
            line.tokens.emplace_back(PossibleAddressToken, ss.str(), a);

            ss.str("");
            ss.clear();
            ss << std::setw(4) << std::setfill('0') << idx + 1;
            line.tokens.emplace_back(TextToken, " │ ID " + ss.str() + ": ");
            ss.str("");
            ss.clear();
            ss << "0x" << std::hex << std::uppercase << b;
            line.tokens.emplace_back(PossibleAddressToken, ss.str(), b);

            ss.str("");
            ss.clear();
            ss << std::setw(4) << std::setfill('0') << idx + 2;
            line.tokens.emplace_back(TextToken, " │ ID " + ss.str() + ": ");
            ss.str("");
            ss.clear();
            ss << "0x" << std::hex << std::uppercase << c;
            line.tokens.emplace_back(PossibleAddressToken, ss.str(), c);

            ss.str("");
            ss.clear();
            ss << std::setw(4) << std::setfill('0') << idx + 3;
            line.tokens.emplace_back(TextToken, " │ ID " + ss.str() + ": ");
            ss.str("");
            ss.clear();
            ss << "0x" << std::hex << std::uppercase << d;
            line.tokens.emplace_back(PossibleAddressToken, ss.str(), d);
            line.tokens.emplace_back(TextToken, " │");

            result.push_back(line);
        }

        int remainingCount = 4 - static_cast<int>(stringOffsets.size() - idx);
        if (remainingCount == 4)
            remainingCount = 0;
        DisassemblyTextLine lineRemainder;
        lineRemainder.addr = addr + (idx * 4);
        for (; idx < stringOffsets.size(); ++idx) {
            ss.str("");
            ss.clear();
            ss << std::setw(4) << std::setfill('0') << idx;
            lineRemainder.tokens.emplace_back(TextToken, "│ ID " + ss.str() + ": ");
            ss.str("");
            ss.clear();
            ss << "0x" << std::hex << std::uppercase << stringOffsets[idx];
            lineRemainder.tokens.emplace_back(PossibleAddressToken, ss.str() + " ", stringOffsets[idx]);
        }
        lineRemainder.tokens.emplace_back(TextToken, "│");
        for (int x = 0; x < remainingCount; x++) {
            ss.str("");
            ss.clear();
            lineRemainder.tokens.emplace_back(TextToken, "                │");
        }
        if (remainingCount != 0)
            result.push_back(lineRemainder);

        DisassemblyTextLine line;
        line.addr = addr + (idx * 4);
        line.tokens.emplace_back(TextToken, "└──────────────────┴──────────────────┴──────────────────┴──────────────────┘");
        result.push_back(line);
    }

    return result;
}