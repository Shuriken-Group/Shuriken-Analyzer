#include "shurikenservice.h"
#include "shurikenservice_internal.h"

std::unique_ptr<ShurikenService> g_shurikenService;

bool ShurikenService::registerView(BinaryNinja::BinaryView *view) {

    try {
        if (m_views.size() > 0) {
            BinaryNinja::LogWarn("ShurikenService::registerView: Only one view is supported");
            return false;
        }
        std::unique_ptr<ShurikenView> shurikenView = std::make_unique<ShurikenView>(view);

        shurikenView->buildMetaData();
        shurikenView->buildSymbols();

        m_views.push_back(std::move(shurikenView));
        return true;
    } catch (const std::exception &e) {
        BinaryNinja::LogInfo("Exception ShurikenService::registerView: %s", e.what());
        return false;
    }
}

const IShurikenView &ShurikenService::getView(BinaryNinja::BinaryView *view) {
    for (const auto &shurikenView: m_views) {
        if (shurikenView->getBinaryView() == view) {
            return std::cref(*shurikenView);
        }
    }
    throw std::runtime_error("View not found");
}

IShurikenService &GetShurikenService() {
    if (!g_shurikenService) {
        g_shurikenService = std::make_unique<ShurikenService>();
    }
    return *g_shurikenService;
}
