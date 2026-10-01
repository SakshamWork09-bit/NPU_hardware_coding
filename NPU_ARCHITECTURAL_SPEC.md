# ==============================================================================
#      DOMAIN-SPECIFIC TENSOR ACCELERATOR (MINI-NPU) ARCHITECTURAL SPECIFICATION
# ==============================================================================
# Author: Saksham Singh Chowhan
# Target Architecture: 2x2 Weight-Stationary Double-Buffered Systolic Processor
# Hardware Description Language: Synthesizable Verilog-2005 (IEEE 1364-2005)
# Verification Toolchain: Icarus Verilog, GTKWave, Yosys Open Synthesis Suite
# ==============================================================================

## 1. Executive Architectural Overview
This document specifies the micro-architecture, silicon resource utilization, 
and verification results for an autonomous, domain-specific 2x2 INT8 Systolic 
NPU Core designed from the ground up in synthesizable RTL.

Unlike general-purpose host CPUs that incur severe thermal throttling (104°C-105°C 
at 68W PL1/PL2 thresholds) due to instruction decode and scheduling logic, this 
accelerator achieves deterministic, cycle-accurate matrix inference utilizing 
only 263 physical D-type flip-flops and 3,627 elementary logic gates.

## 2. Micro-Architectural Subsystems

### A. Double-Buffered Processing Element (PE)
- Arithmetic: 1-cycle signed INT8 x INT8 MAC with a 32-bit accumulator.
- Ping-Pong Register Storage:
  * Bank 0 (Active Register): Locks weights feeding the multiplier during inference.
  * Bank 1 (Shadow Register): Concurrently buffers upcoming layer weights via `load_weight`.
- Zero-Stall Swap: Strobe `swap_weights` flips banks in 1 clock cycle without dropping 
  pipeline throughput.

### B. Hardware Wavefront Skew Buffers
- Offloads input staggering from external software onto silicon shift registers.
- Row 0 routes directly to PE[0][0] (0-cycle delay).
- Row 1 passes through an internal 8-bit DFF buffer, creating the mandatory 
  1-cycle diagonal wavefront delay required for systolic partial-sum alignment.

### C. Autonomous Hardware Controller (Finite State Machine)
- Replaces manual external testbench control with on-chip silicon autonomy.
- State Machine Sequence:
  * S_IDLE (000)   -> Listens for host pulse `start`.
  * S_LOAD_W (001) -> Captures layer weights into shadow registers.
  * S_SWAP_W (010) -> Executes zero-stall bank flip into active compute registers.
  * S_STREAM (011) -> Sequences matrix inputs across skew buffers.
  * S_DRAIN (100)  -> Flushes systolic wavefront through PE[1][1].
  * S_DONE (101)   -> Fires 1-cycle interrupt pulse `done` to alert host CPU.

### D. Pipelined Re-Quantization & Activation Stage
- Non-linear Activation: Clamps negative accumulators to zero: ReLU(x) = max(0, x).
- Fixed-Point Scaling: Arithmetic right shift (>>> SHIFT_BITS) normalizes wide accumulators.
- INT8 Saturation: Clamps overflow values to 127, outputting standard signed 8-bit tensors.

## 3. Physical Gate & Register Footprint (Yosys Synthesis Netlist)

- Top-Level Module: npu_top
- Technology Primitives: Elementary CMOS Standard Cells

| Primitive Type | Synthesized Cell Description | Quantity |
| :--- | :--- | :--- |
| `$_SDFFE_PN0P_` | D-Flip-Flop with Clock Enable & Synchronous Reset | 227 |
| `$_SDFF_PN0_`   | D-Flip-Flop with Synchronous Reset (Active Low)   | 34 |
| `$_SDFF_PP0_`   | D-Flip-Flop with Synchronous Reset (Active High)  | 2 |
| **TOTAL REGISTERS** | **Total Physical D-Type Flip-Flops**          | **263** |
| `$_AND_`        | 2-Input CMOS AND Gate (Partial Products)          | 1,551 |
| `$_OR_`         | 2-Input CMOS OR Gate (Carry Lookahead Logic)      | 628 |
| `$_XOR_`        | 2-Input CMOS XOR Gate (Full Adder Trees)          | 911 |
| `$_MUX_`        | 2-to-1 Multiplexer (Data Routing Networks)        | 101 |
| `$_NOT_`        | Inverter Cell                                     | 173 |
| **TOTAL CELLS** | **Total Silicon Gate Primitives**                 | **3,627** |

## 4. Verification Milestone Audit

1. Functional Win:
   Multiplication of Matrix A [[1, 2], [3, 4]] by W [[2, 3], [4, 5]] verified:
   - Raw 32-bit Outputs: C00=14, C10=20, C01=18, C11=26.
   - Post-Quantization Outputs (>>> 1): Out00=7, Out10=10, Out01=9, Out11=13.
2. Timing Win:
   - Zero pipeline bubbles across matrix boundaries.
   - Handshake flags (`busy`, `done`, `out_valid`) verified under GTKWave inspection.
3. Gate Synthesis Win:
   - Full gate-level lowering confirmed via Yosys with 0 inferred latches.
# ==============================================================================
