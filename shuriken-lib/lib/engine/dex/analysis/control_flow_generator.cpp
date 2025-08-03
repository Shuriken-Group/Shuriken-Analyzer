//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/engine/dex/analysis/control_flow_generator.hpp"
#include "shuriken/internal/engine/dex/disassembler/internal_disassembler.hpp"
#include "shuriken/internal/sdk/dex/instruction_impl.hpp"
#include "shuriken/internal/sdk/dex/method_impl.hpp"

using namespace shuriken::dex;

std::unique_ptr<ControlFlowGraph> ControlFlowGeneratorPass::generate_control_flow_graph(Method::Impl* method_impl) {
    // Initialize the control flow graph
    current_cfg_impl = new ControlFlowGraph::Impl();
    current_cfg_ = std::make_unique<ControlFlowGraph>(current_cfg_impl);
    address_to_block_.clear();

    // Check if method has instructions
    if (!method_impl || method_impl->get_method_instructions().size() == 0) {
        return std::move(current_cfg_);
    }

    // Create basic blocks from instruction sequence
    create_basic_blocks(method_impl);
    
    // Build control flow edges between blocks
    build_control_flow_edges(method_impl);
    
    // Handle exception flow
    handle_exception_flow(method_impl);

    return std::move(current_cfg_);
}

void ControlFlowGeneratorPass::create_basic_blocks(Method::Impl* method_impl) {
    // Find basic block entry points
    auto entry_points = find_basic_block_leaders(method_impl);
    
    if (entry_points.empty()) {
        // Single basic block for the entire method
        auto & last_instr = method_impl->get_method_instructions().back();
        auto* block = create_basic_block(0, last_instr.get().get_address(), method_impl);
        current_cfg_impl->add_node(block);
        return;
    }

    std::int64_t start = 0, end = 0;
    DVMBasicBlock* current = nullptr;
    DVMBasicBlock* prev = nullptr;

    for (auto & instruction : method_impl->get_method_instructions()) {
        auto idx = instruction.get().get_address();

        // Check if we're at a basic block entry point
        if (entry_points.find(idx) != entry_points.end()) {
            prev = current;
            
            // Create new basic block
            current = create_basic_block(start, end, method_impl);
            
            // Add first block or handle fallthrough
            if (start == 0) {
                current_cfg_impl->add_node(current);
            } else if (prev != nullptr) {
                // Check for fallthrough edge
                if (!prev->get_terminator()->is_terminator()) {
                    current_cfg_impl->add_edge(prev, current);
                } else {
                    current_cfg_impl->add_node(current);
                }
            }
            
            start = end + (current ? current->get_terminator()->get_instruction_length() : 0);
        }
        
        end = instruction.get().get_address();
    }

    // Handle final basic block
    if (current == nullptr) {
        current = create_basic_block(start, end, method_impl);
        current_cfg_impl->add_node(current);
    } else if (start != end || current_cfg_->get_basic_block_by_idx(start) == nullptr) {
        prev = current;
        current = create_basic_block(start, end, method_impl);
        
        if (prev && !prev->get_terminator()->is_terminator()) {
            current_cfg_impl->add_edge(prev, current);
        } else {
            current_cfg_impl->add_node(current);
        }
    }
}

void ControlFlowGeneratorPass::build_control_flow_edges(Method::Impl* method_impl) {
    shuriken::dex::InternalDisassembler disassembler;
    
    // Build jump target map
    std::unordered_map<std::uint64_t, std::vector<std::int64_t>> target_jumps;
    
    for (auto& instruction : method_impl->get_method_instructions()) {
        if (instruction.get().is_jump_instruction()) {
            auto idx = instruction.get().get_address();
            auto targets = disassembler.determine_next(&instruction.get(), idx);
            target_jumps[idx] = std::move(targets);
        }
    }

    // Get method bounds
    auto& last_instr = method_impl->get_method_instructions().back();
    auto out_range = last_instr.get().get_address() + last_instr.get().get_instruction_length();

    // Add jump edges
    for (const auto& jump_target : target_jumps) {
        auto src_idx = jump_target.first;
        auto* src = current_cfg_impl->get_basic_block_by_idx(src_idx);
        
        if (src_idx >= out_range || src == nullptr) {
            continue;
        }

        for (auto dst_idx : jump_target.second) {
            auto* dst = current_cfg_impl->get_basic_block_by_idx(dst_idx);
            
            if (dst_idx >= static_cast<std::int64_t>(out_range) || dst == nullptr) {
                continue;
            }
            
            current_cfg_impl->add_edge(src, dst);
        }
    }
}

void ControlFlowGeneratorPass::handle_exception_flow(Method::Impl* method_impl) {
    // Handle exception handlers
    for (const auto& except : method_impl->get_exceptions()) {
        for (const auto& handler : except.handler) {
            auto* catch_bb = current_cfg_->get_basic_block_by_idx(handler.handler_start_addr);
            if (catch_bb == nullptr) {
                // Create basic block for exception handler
                auto * catch_bb_impl = new DVMBasicBlock::Impl(handler.handler_start_addr, handler.handler_start_addr + 1);
                catch_bb = new DVMBasicBlock(catch_bb_impl);
                block_to_implementation_[catch_bb] = catch_bb_impl;
                current_cfg_impl->add_node(catch_bb);
            }
        }
    }

    // Add exception edges
    for (const auto& except : method_impl->get_exceptions()) {
        auto* try_bb = current_cfg_->get_basic_block_by_idx(except.try_value_start_addr);
        auto* try_bb_impl = block_to_implementation_[try_bb];
        if (try_bb_impl != nullptr) {
            try_bb_impl->set_is_try_block(true);

            for (const auto& handler : except.handler) {
                auto* catch_bb = current_cfg_->get_basic_block_by_idx(handler.handler_start_addr);
                auto* catch_bb_impl = block_to_implementation_[catch_bb];
                if (catch_bb_impl != nullptr) {
                    try_bb_impl->add_catch_block(catch_bb);
                    catch_bb_impl->set_is_catch_block(true);
                    catch_bb_impl->add_handler(handler.handler_data);
                }
            }
        }
    }
}

std::set<std::uint64_t> ControlFlowGeneratorPass::find_basic_block_leaders(Method::Impl* method_impl) {
    std::set<std::uint64_t> entry_points;
    shuriken::dex::InternalDisassembler disassembler;

    // Find jump targets

    for (auto & instruction : method_impl->get_method_instructions()) {
        if (instruction.get().is_jump_instruction()) {
            auto idx = instruction.get().get_address();
            auto targets = disassembler.determine_next(&instruction.get(), idx);
            
            for (auto target : targets) {
                entry_points.insert(target);
            }
        }
    }

    // Find exception handler entry points
    for (const auto& except : method_impl->get_exceptions()) {
        for (const auto& handler : except.handler) {
            entry_points.insert(handler.handler_start_addr);
        }
    }

    // Remove invalid entry points
    entry_points.erase(0);

    return entry_points;
}

DVMBasicBlock* ControlFlowGeneratorPass::create_basic_block(std::uint64_t start_address, std::uint64_t end_address, Method::Impl* method_impl) {
    // Get instruction span for this basic block
    auto instruction_span = method_impl->get_instructions_in_range(start_address, end_address);
    
    // Create the basic block
    auto* impl = new DVMBasicBlock::Impl(instruction_span);
    auto* block = new DVMBasicBlock(impl);
    block_to_implementation_[block] = impl;
    
    // Store in address mapping
    address_to_block_[start_address] = block;
    
    return block;
}