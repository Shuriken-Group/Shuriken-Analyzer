
project(MJOLNIR)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Set this so we dont run into problems including help scripts from mlir and llvm
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include_directories(SYSTEM "${MLIR_INCLUDE_DIRS}")

include(AddMLIR)
include(AddLLVM)
include(TableGen)


# Set up necessary llvm definitions and libraries
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS_LIST})
llvm_map_components_to_libnames(llvm_libs native orcjit core support irreader option)

message("MLIR dir is ${MLIR_DIR}")
message(${CMAKE_CURRENT_SOURCE_DIR})
message(${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)


add_executable(shuriken-opt src/shuriken_opt.cpp)
target_link_libraries(shuriken-opt
  PRIVATE
  ${llvm_libs} LLVMSupport
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
add_dependencies(shuriken-opt ShurikenTblGen)
target_include_directories(shuriken-opt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/mjolnir/)
target_include_directories(shuriken-opt PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include/mjolnir/)

llvm_update_compile_flags(shuriken-opt)
