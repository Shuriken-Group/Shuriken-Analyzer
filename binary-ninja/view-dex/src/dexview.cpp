#include "plugin.h"
#undef SHURIKENSERVICE_EXPORTS
#include "shurikenservice.h"

// Ugly hack for proof of concept ----------------------------------------------
typedef void (*SetBinaryView)(BinaryNinja::BinaryView *bv);

void ArchBVHack(BinaryNinja::BinaryView *bv) {
    // windows
    HMODULE hModule = LoadLibraryA("arch-dalvik.dll");
    SetBinaryView setBV = (SetBinaryView) GetProcAddress(hModule, "SetBinaryView");
    setBV(bv);

    // Linux

    // Mac
}
// ------------------------------------------------------------------------------

namespace BinaryNinja {

    DEXView::DEXView(BinaryView *data, bool parseOnly) : BinaryView("DEX", data->GetFile(), data), m_parseOnly(parseOnly) {
        CreateLogger("BinaryView");
        m_logger = CreateLogger("BinaryView.DEXView");
        m_backedByDatabase = data->GetFile()->IsBackedByDatabase("DEX");
    }

    bool DEXView::Init() {
        m_logger->LogError("DEXView::Init()");
        Ref<Settings> settings = GetLoadSettings(GetTypeName());
        Ref<Settings> viewSettings = Settings::Instance();

        const uint64_t alignment = 0x1000;
        const uint64_t rawFileOffset = 0;
        const uint64_t dexCodeSegmentSize = (GetParentView()->GetLength() + alignment - 1) & ~(alignment - 1);
        const uint64_t fieldDataSegmentAddress = m_imageBase + dexCodeSegmentSize;
        const uint64_t fieldDataSegmentSize = 0x1000;

        m_arch = Architecture::GetByName("Dalvik");
        SetDefaultArchitecture(m_arch);

        m_platform = m_arch->GetStandalonePlatform();
        SetDefaultPlatform(m_platform);

        // TODO: proper segment and sections creation
        AddAutoSegment(m_imageBase, GetParentView()->GetLength(), rawFileOffset, GetParentView()->GetLength(), SegmentReadable | SegmentExecutable | SegmentContainsCode);
        AddAutoSection("raw", m_imageBase, GetParentView()->GetLength(), ReadOnlyCodeSectionSemantics);

        IShurikenService &shurikenService = GetShurikenService();
        if (!shurikenService.registerView(this))
            return false;
        // arch plugin is global, so we need to set the current view
        ArchBVHack(this);

        return true;
    }

    uint64_t DEXView::PerformGetEntryPoint() const {
        return 0;
    }

    size_t DEXView::PerformGetAddressSize() const {
        return 8;
    }

}// namespace BinaryNinja