# Synthesizable 2x2 Systolic NPU Core & Real-Time HIL Vision Accelerator

A high-efficiency, synthesizable 2x2 weight-stationary systolic NPU core mapped to the SkyWater 130nm CMOS node.

## Silicon Implementation Summary
- **Target Process**: SkyWater 130nm CMOS Standard Cell (Yosys / ABC)
- **Total Standard Cells**: 2,526 Gates
- **Sequential Elements**: 185 D-Flip-Flops
- **Silicon F_max**: 305.34 MHz
- **Peak Arithmetic Density**: 2.443 GOPS (INT8 MACs)
- **Setup Slack @ 100MHz**: +6.725 ns (MET)
- **Core Area**: 200 um x 200 um (0.04 mm²)

## Architecture
- Double-buffered weight-stationary processing elements (0-stall layer swaps)
- Integrated DFF wavefront skew buffer on Row 1
- Autonomous 6-state FSM controller (IDLE -> LOAD -> SWAP -> STREAM -> DRAIN -> DONE)
- Combinational in-pipeline requantization (ReLU + INT8 saturation)
- Live hardware-in-the-loop (HIL) camera pipeline and web telemetry dashboard
