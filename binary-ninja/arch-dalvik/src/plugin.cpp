#include "plugin.h"

using namespace BinaryNinja;

extern "C" {
BN_DECLARE_CORE_ABI_VERSION

BINARYNINJAPLUGIN bool CorePluginInit() {

    BinaryNinja::Architecture* archDalvik = new DalvikArchitecture("Dalvik", LittleEndian);
    Architecture::Register(archDalvik);

    return true;
}
}