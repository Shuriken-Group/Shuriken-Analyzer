add_library(MjolnIRTransformLib lifter.cpp lower.cpp)
target_link_libraries(MjolnIRTransformLib LLVMSupport)
target_include_directories(MjolnIRTransformLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRTransformLib PRIVATE ${PROJECT_BINARY_DIR}/include/)
