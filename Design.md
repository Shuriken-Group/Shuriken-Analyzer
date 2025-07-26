# Shuriken Analyzer Architecture Design

## Overview

Shuriken Analyzer uses a modern PIMPL (Pointer to Implementation) pattern that provides a clean public API while hiding implementation details. This architecture reduces memory usage by 50% compared to the previous Provider pattern by eliminating duplicate object storage.

## New PIMPL Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  USER CODE                                      │
│                              (Applications)                                     │
└─────────────────────────────────────────────────────────────┬───────────────────┘
                                                              │
┌─────────────────────────────────────────────────────────────▼───────────────────┐
│                          SDK LAYER (Public Interface)                           │
│                                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │   Dex   │  │  Class  │  │ Method  │  │  Field  │  │External │  │External │   │
│  │         │  │         │  │         │  │         │  │ Method  │  │  Field  │   │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │   │
│  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │  │ │Impl*│ │   │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │ owns       │ owns       │ owns       │ owns       │ owns       │ owns   │
│       │ unique_ptr │ unique_ptr │ unique_ptr │ unique_ptr │ unique_ptr │unique  │
└───────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────┘
        │            │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION LAYER (PIMPL)                              │
│                      (Hidden Implementation Details)                          │
│                                                                               │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│ │   Dex    │ │  Class   │ │  Method  │ │  Field   │ │External  │ │External │  │
│ │   Impl   │ │   Impl   │ │   Impl   │ │   Impl   │ │ Method   │ │  Field  │  │
│ │          │ │          │ │          │ │          │ │   Impl   │ │  Impl   │  │
│ │ • Parser │ │ • Name   │ │ • Name   │ │ • Name   │ │ • Name   │ │ • Name  │  │
│ │ • Engine │ │ • Package│ │ • Flags  │ │ • Type   │ │ • Class  │ │ • Class │  │
│ │ • Pools  │ │ • Methods│ │ • Class  │ │ • Class  │ │ • Desc   │ │ • Desc  │  │
│ │ • Storage│ │ • Fields │ │ • Code   │ │ • XRefs  │ │          │ │         │  │
│ │ • Maps   │ │ • XRefs  │ │ • XRefs  │ │          │ │          │ │         │  │
│ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └────┬────┘  │
│       │            │            │            │            │           │       │
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

## Key Components

### SDK Layer (Public API)
- **Clean Interface**: Users only interact with SDK classes
- **PIMPL Pattern**: Each SDK object owns its implementation via `std::unique_ptr<Impl>`
- **Type Safety**: Strong typing with custom type aliases
- **Binary Compatibility**: Implementation changes don't break ABI
- **Direct Ownership**: No reference wrapper overhead

### Implementation Layer (PIMPL)
- **Hidden Details**: Implementation classes contain actual data and business logic
- **Encapsulation**: Private nested `Impl` classes keep internals hidden
- **Memory Efficiency**: Single allocation per object (SDK + Impl combined)
- **Direct Access**: No indirection through provider layer

### Engine Layer (Embedded in Dex::Impl)
- **Integrated Design**: Parser and engine embedded directly in `Dex::Impl`
- **Centralized Storage**: All objects stored in `Dex::Impl` containers
- **Efficient Lookup**: Fast ID mapping tables for object resolution
- **Pool Management**: Handles string, type, and prototype pools

## Data Flow

1. **File Loading**: DEX binary file is loaded into ShurikenStream
2. **Parsing**: `Dex::Impl` parser extracts structured data from binary format
3. **Object Creation**: `Dex::Impl` creates SDK objects with embedded implementations
4. **Direct Storage**: Objects stored directly in `Dex::Impl` containers
5. **User Access**: Applications use clean SDK interface

## Memory Optimization: 50% Reduction

### Old Provider Pattern (2 objects per entity):
```
Class object → DexClassProvider object → Data
Method object → DexMethodProvider object → Data
Field object → DexFieldProvider object → Data
```

### New PIMPL Pattern (1 object per entity):
```
Class object (contains Class::Impl) → Data
Method object (contains Method::Impl) → Data
Field object (contains Field::Impl) → Data
```

**Result**: Eliminated intermediate Provider objects = 50% fewer allocations

## External Reference Handling

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

- **Internal Objects**: Full implementation with bytecode, instructions, etc.
- **External Objects**: Basic metadata only (name, descriptor, class)
- **Lazy Creation**: External objects created when referenced
- **Unified Interface**: Same API for both internal and external objects

## Benefits

- **Separation of Concerns**: Clear boundaries between layers
- **Maintainability**: Implementation changes don't affect public API
- **Performance**: Reference caches and lazy loading optimize memory
- **Extensibility**: Easy to add new object types or analysis features
- **Safety**: PIMPL pattern hides implementation complexity