#pragma once

#include "binaryninjaapi.h"
#include "shurikenservice.h"
#include "shurikenservice_internal.h"
#include <memory>
#include <string_view>


class ShurikenView : public IShurikenView {

public:
    explicit ShurikenView(BinaryNinja::BinaryView *bv);

    void buildMetaData();
    void buildSymbols();
    BinaryNinja::BinaryView *getBinaryView() { return m_bv; };

    const shurikenapi::IDex &getParser() const override;
    uint32_t getStringOffset(int64_t id) const override;

private:
    BinaryNinja::Ref<BinaryNinja::Function> buildMethod(const shurikenapi::IClassMethod &method);
    BinaryNinja::Ref<BinaryNinja::Type> getFundamental(const shurikenapi::FundamentalValue &value);

    BinaryNinja::BinaryView *m_bv;
    std::unique_ptr<shurikenapi::IDex> m_parser;
    std::unordered_map<int64_t, uint32_t> m_stringMap;
};

class ShurikenService : public IShurikenService {
public:
    ShurikenService() = default;
    bool registerView(BinaryNinja::BinaryView *view) override;
    const IShurikenView &getView(BinaryNinja::BinaryView *view) override;

private:
    std::vector<std::unique_ptr<ShurikenView>> m_views;
};
