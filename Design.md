# Shuriken Analyzer Architecture Design

## Overview

Shuriken Analyzer is a high-performance C++ library for Android DEX bytecode analysis using a modern PIMPL (Pointer to Implementation) pattern. The architecture provides a clean public API while hiding implementation details, achieving 50% memory reduction compared to previous provider-based designs.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  USER CODE                                      │
│                              (Applications)                                     │
└─────────────────────────────────────────────────────────┬───────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────▼───────────────────────┐
│                          SDK LAYER (Public Interface)                           │
│                                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │   Dex   │  │  Class  │  │ Method  │  │  Field  │  │External │  │Control  │   │
│  │         │  │         │  │         │  │         │  │Objects  │  │Flow     │   │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │   │
│  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │   │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │ owns       │ owns       │ owns       │ owns       │ owns       │ owns   │
│       │unique_ptr  │unique_ptr  │unique_ptr  │unique_ptr  │unique_ptr  │unique  │
└───────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────┘
        │            │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION LAYER (PIMPL)                              │
│                      (Hidden Implementation Details)                          │
│                                                                               │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│ │   Dex    │ │  Class   │ │  Method  │ │  Field   │ │External  │ │Control  │  │
│ │   Impl   │ │   Impl   │ │   Impl   │ │   Impl   │ │ Objects  │ │Flow     │  │
│ │          │ │          │ │          │ │          │ │   Impl   │ │  Impl   │  │
│ │ • Parser │ │ • Name   │ │ • Name   │ │ • Name   │ │ • Name   │ │ • Nodes │  │
│ │ • Engine │ │ • Package│ │ • Flags  │ │ • Type   │ │ • Class  │ │ • Edges │  │
│ │ • Pools  │ │ • Methods│ │ • Class  │ │ • Class  │ │ • Desc   │ │ • Graph │  │
│ │ • Storage│ │ • Fields │ │ • Code   │ │ • XRefs  │ │          │ │         │  │
│ │ • Maps   │ │ • XRefs  │ │ • XRefs  │ │          │ │          │ │         │  │
│ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └────┬────┘  │
│       │            │            │            │            │           │       │
│       │            └────────────┼────────────┼────────────┼───────────┼───────┤
│       │                         │            │            │           │       │
│       │                         ▼            ▼            ▼           ▼       │
│       │                    ┌─────────────────────────────────────────────┐    │
│       │                    │         Centralized Object Storage          │    │
│       │                    │            (In Dex::Impl)                   │    │
│       │                    │                                             │    │
│       │                    │  • std::vector<unique_ptr<Class>>           │    │
│       │                    │  • std::vector<unique_ptr<Method>>          │    │
│       │                    │  • std::vector<unique_ptr<Field>>           │    │
│       │                    │  • std::vector<unique_ptr<ExternalMethod>>  │    │
│       │                    │  • std::vector<unique_ptr<ExternalField>>   │    │
│       │                    │  • Reference caches for fast iteration      │    │
│       │                    │  • ID mapping tables (MethodID->Object)     │    │
│       │                    └─────────────────────────────────────────────┘    │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │                      DEX PARSING ENGINE                             │      │
│  │                      (Embedded in Dex::Impl)                        │      │
│  │                                                                     │      │
│  │  ┌─────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │      │
│  │  │ Parser  │  │Disassembler │  │String Pool  │  │  Type Pool  │     │      │
│  │  │         │  │             │  │             │  │             │     │      │
│  │  │         │  │             │  │             │  │             │     │      │
│  │  └─────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │      │
│  └─────────────────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                 DEX FILE                                      │
│                              (Binary Format)                                  │
│                                                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Header  │  │ String  │  │  Type   │  │ Proto   │  │ Method  │  │  Field  │ │
│  │         │  │  IDs    │  │  IDs    │  │  IDs    │  │  IDs    │  │  IDs    │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
│                                                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                           │
│  │ Class   │  │ Code    │  │ String  │  │  Data   │                           │
│  │  Defs   │  │ Items   │  │  Data   │  │Section  │                           │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                           │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SDK Layer (Public API) - `shuriken/sdk/`

**Main Entry Point:**
- `Dex` - Primary interface for DEX file analysis

**Core Object Model:**
- `Class` - Represents Java/Kotlin classes with methods, fields, inheritance
- `Method` - Methods with bytecode, instructions, control flow analysis
- `Field` - Class fields with type information and cross-references
- `Instruction` - Dalvik bytecode instructions (35+ format types)

**Analysis Components:**
- `DVMBasicBlock` - Basic blocks for control flow analysis
- `ControlFlowGraph` - Control flow representation with nodes and edges
- `DVMTypes` - Comprehensive type system (fundamental, class, array)
- `DVMPrototype` - Method signatures and prototypes

**External References:**
- `ExternalMethod` - Methods from external DEX files
- `ExternalField` - Fields from external DEX files

**Utilities:**
- `iterator_range` - Efficient range-based iteration
- `ShurikenStream` - Cross-platform file I/O abstraction
- Type aliases for type safety and clarity

### 2. Implementation Layer - `shuriken/internal/`

**PIMPL Classes:**
Every SDK class has a corresponding private `Impl` class:
```cpp
class Class {
public:
    class Impl;  // Forward declared implementation
private:
    std::unique_ptr<Impl> pimpl;  // Hidden implementation
};
```

**Implementation Classes contain:**
- Actual data storage and business logic
- Private methods and implementation details
- Direct access to parser and engine components
- Memory-efficient object storage

### 3. Engine Layer - `DexEngine`

**Integrated Design:**
- Embedded directly in `Dex::Impl` for performance
- Single-pass parsing with efficient memory layout
- Centralized object storage and management

**Core Engine Components:**
- **Parser**: DEX file format parsing and validation
- **Disassembler**: Bytecode to instruction object conversion  
- **Pool Management**: String, type, and prototype pools
- **Object Factory**: Creates and manages all SDK objects
- **Cross-Reference Builder**: Builds method/field/class relationships

### 4. Advanced Instruction System

**Comprehensive Format Support:**
- 35+ specific instruction formats (10x, 11n, 21c, 35c, etc.)
- Format-specific operand access methods
- Switch statement support (`PackedSwitch`, `SparseSwitch`)

**Analysis Features:**
- Side effect detection (read/write/call/throw)
- Terminator instruction identification
- Exception throwing instruction detection
- Control flow transfer analysis

### 5. Control Flow Analysis

**DVMBasicBlock Features:**
- Instruction sequence management
- Address range tracking
- Predecessor/successor relationships
- String representation for debugging

**ControlFlowGraph Features:**
- Node and edge management with ownership
- Iterator-based traversal
- Predecessor/successor queries
- Graph visualization support

### 6. Type System

**DVMType Variant System:**
```cpp
using DVMType = std::variant<DVMFundamental*, DVMClass*, DVMArray*>;
```

**Type Categories:**
- **DVMFundamental**: Primitive types (int, boolean, etc.)
- **DVMClass**: Reference types with full metadata
- **DVMArray**: Array types with dimension and element type

**Access Flags:**
- Complete Java/Dalvik access modifier support
- Type-safe enum representations
- Bitwise operation support

## Key Design Patterns

### PIMPL (Pointer to Implementation)

**Benefits:**
- **50% Memory Reduction**: Eliminated duplicate Provider objects
- **ABI Stability**: Implementation changes don't break binary compatibility
- **Encapsulation**: Complete hiding of implementation details
- **Performance**: Direct object ownership without indirection

**Memory Comparison:**
```cpp
// Old Provider Pattern (2 objects per entity)
Class object → DexClassProvider object → Data

// New PIMPL Pattern (1 object per entity)
Class object (contains Class::Impl) → Data
```

### Factory Pattern

**Object Creation:**
- `Dex::Impl` serves as the central factory
- Creates all objects during parsing phase
- Maintains object ownership and lifecycle
- Provides fast ID-to-object mapping

### Iterator Pattern

**Range-Based Access:**
```cpp
for (auto& method : dex_class.get_methods()) {
    // Process method
}

for (auto& instruction : method.get_instructions()) {
    // Process instruction
}
```

## Data Flow Architecture

1. **File Loading**: DEX binary loaded via `ShurikenStream`
2. **Parsing**: `DexEngine` parses binary format into structured data
3. **Object Creation**: Factory creates SDK objects with embedded implementations
4. **Storage**: Objects stored in centralized `Dex::Impl` containers
5. **Analysis**: Control flow graphs and cross-references built
6. **Access**: Applications use clean SDK interface for analysis

## Cross-Reference System

**Comprehensive Tracking:**
- **Class References**: Inheritance, implementation, usage relationships
- **Method Calls**: Caller/callee relationships with call sites
- **Field Access**: Read/write operations with access locations
- **Type Usage**: Instance creation, type casting, annotations

**External Reference Handling:**
```
Internal DEX File          External DEX File
┌─────────────┐            ┌─────────────┐
│    Class A  │            │    Class C  │
│             │            │             │
│  Method M1  │──calls──►  │  Method M3  │
│             │            │             │
│  Field F1   │──reads──►  │  Field F3   │
└─────────────┘            └─────────────┘
      │                           │
      ▼                           ▼
┌─────────────┐            ┌─────────────┐
│   Method    │            │ External    │
│   Object    │            │ Method      │
│ (Full Data) │            │ (Metadata)  │
└─────────────┘            └─────────────┘
```

## Performance Optimizations

### Memory Efficiency
- **Single Allocation**: Combined SDK object + implementation
- **Reference Caches**: Fast iteration without object creation overhead
- **Lazy Loading**: External objects created only when needed
- **Pool Sharing**: Shared string and type pools across all objects

### CPU Performance
- **Zero-Copy**: Direct span access to bytecode data
- **Fast Lookups**: Hash maps for ID-to-object resolution
- **Iterator Ranges**: Efficient range-based access patterns
- **Minimal Indirection**: Direct pointer access in PIMPL

### Cross-Platform Support
- **Standard C++20**: No platform-specific dependencies
- **CMake Build System**: Portable build configuration
- **Cross-Platform I/O**: Abstracted file operations
- **WebAssembly Ready**: Minimal std::filesystem usage

## Build System Architecture

**Modular CMake Design:**
```
CMakeLists.txt (root)
├── shuriken-lib/
│   ├── CMakeLists.txt (library configuration)
│   └── lib/sdk/dex/CMakeLists.txt (component-specific)
├── shuriken-dump/
│   └── CMakeLists.txt (CLI tool)
└── externals/
    └── CMakeLists.txt (dependencies)
```

**Key Features:**
- **Object Library**: `LIB_SHURIKEN` for efficient compilation
- **Export System**: CMake package configuration for consumers
- **Development Mode**: Optional internal header exposure
- **Dependency Management**: FetchContent for external libraries

## Extension Points

### Adding New Instruction Formats
1. Inherit from `Instruction` base class
2. Implement format-specific operand access
3. Register with instruction factory
4. Add disassembly support

### Custom Analysis Passes
1. Iterate over control flow graphs
2. Access instruction metadata
3. Build custom data structures
4. Integrate with cross-reference system

### External Tool Integration
1. Use public SDK interface
2. Access parsed objects via iterators
3. Build analysis results
4. Export to desired formats

## Benefits

- **Clean Architecture**: Clear separation between public API and implementation
- **High Performance**: Optimized memory layout and access patterns
- **Maintainability**: PIMPL pattern enables safe refactoring
- **Extensibility**: Factory pattern supports new object types
- **Cross-Platform**: Standard C++ with minimal dependencies
- **Memory Efficient**: 50% reduction compared to previous designs
- **Type Safe**: Strong typing with custom type aliases
- **Analysis Ready**: Built-in control flow and cross-reference analysis