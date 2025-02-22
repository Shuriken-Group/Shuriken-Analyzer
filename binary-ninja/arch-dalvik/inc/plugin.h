#pragma once

#include "binaryninjaapi.h"
#include <string>
#include <vector>

class DalvikArchitecture : public BinaryNinja::Architecture {
  public:
    DalvikArchitecture(const char* name, BNEndianness endian);

    BNEndianness GetEndianness() const override;
    size_t GetAddressSize() const override;
    size_t GetMaxInstructionLength() const override;
    size_t GetDefaultIntegerSize() const override;
    size_t GetInstructionAlignment() const override;

    bool GetInstructionInfo(const uint8_t* data, uint64_t addr, size_t maxLen, BinaryNinja::InstructionInfo& result) override;
    bool GetInstructionText(const uint8_t* data, uint64_t addr, size_t& len,
                            std::vector<BinaryNinja::InstructionTextToken>& result) override;
    //bool GetInstructionLowLevelIL(const uint8_t* data, uint64_t addr, size_t& len, BinaryNinja::LowLevelILFunction& il) override;

  private:
    BNEndianness m_endian;
};