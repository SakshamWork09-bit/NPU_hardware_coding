`timescale 1ns / 1ps

module npu_top (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    
    // Weight Bus
    input  wire signed [7:0]  w00, w01, w10, w11,
    
    // Input Matrix Activations (INT8)
    input  wire signed [7:0]  a_col0_row0, a_col0_row1,
    input  wire signed [7:0]  a_col1_row0, a_col1_row1,

    // Rescaled Output Activation Stream (INT8 Ready for Next Layer)
    output wire signed [7:0]  out_col0_int8,
    output wire signed [7:0]  out_col1_int8,
    output wire               out_valid_col0,
    output wire               out_valid_col1,
    output wire               busy,
    output wire               done
);

    wire signed [31:0] raw_c_col0, raw_c_col1;
    wire               raw_val_col0, raw_val_col1;

    // Instantiate Autonomous Compute Core
    npu_core_2x2 core (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .a_col0_row0(a_col0_row0), .a_col0_row1(a_col0_row1),
        .a_col1_row0(a_col1_row0), .a_col1_row1(a_col1_row1),
        .c_out_col0(raw_c_col0),
        .c_out_col1(raw_c_col1),
        .valid_out_col0(raw_val_col0),
        .valid_out_col1(raw_val_col1),
        .busy(busy),
        .done(done)
    );

    // Column 0 Post-Processing Pipeline
    requant_unit #(.SHIFT_BITS(1)) post0 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(raw_val_col0),
        .acc_in(raw_c_col0),
        .valid_out(out_valid_col0),
        .data_out(out_col0_int8)
    );

    // Column 1 Post-Processing Pipeline
    requant_unit #(.SHIFT_BITS(1)) post1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(raw_val_col1),
        .acc_in(raw_c_col1),
        .valid_out(out_valid_col1),
        .data_out(out_col1_int8)
    );

endmodule
