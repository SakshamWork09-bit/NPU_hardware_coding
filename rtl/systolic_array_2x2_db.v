`timescale 1ns / 1ps

module systolic_array_2x2_db (
    input  wire               clk,
    input  wire               rst_n,

    // Double-buffered weight control interface
    input  wire               load_weight,   // Pulse high to load shadow registers
    input  wire               swap_weights,  // 1-cycle strobe: shadow -> active registers
    input  wire signed [7:0]  w00,
    input  wire signed [7:0]  w01,
    input  wire signed [7:0]  w10,
    input  wire signed [7:0]  w11,

    // Streaming activation inputs (West boundary)
    input  wire               valid_in_row0,
    input  wire               valid_in_row1,
    input  wire signed [7:0]  a_in_row0,
    input  wire signed [7:0]  a_in_row1,

    // Matrix outputs (South boundary)
    output wire signed [31:0] c_out_col0,
    output wire signed [31:0] c_out_col1,
    output wire               valid_out_col0,
    output wire               valid_out_col1
);

    // Internal inter-PE systolic routing
    wire signed [7:0]  a_00_to_01;
    wire signed [7:0]  a_10_to_11;
    wire signed [31:0] acc_00_to_10;
    wire signed [31:0] acc_01_to_11;
    wire               val_00_to_01;
    wire               val_10_to_11;

    // PE[0][0] - Top Left
    pe_double_buffered pe00 (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .weight_in(w00),
        .valid_in(valid_in_row0),
        .a_in(a_in_row0),
        .acc_in(32'sd0),
        .a_out(a_00_to_01),
        .acc_out(acc_00_to_10),
        .valid_out(val_00_to_01)
    );

    // PE[0][1] - Top Right
    pe_double_buffered pe01 (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .weight_in(w01),
        .valid_in(val_00_to_01),
        .a_in(a_00_to_01),
        .acc_in(32'sd0),
        .a_out(),
        .acc_out(acc_01_to_11),
        .valid_out()
    );

    // PE[1][0] - Bottom Left
    pe_double_buffered pe10 (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .weight_in(w10),
        .valid_in(valid_in_row1),
        .a_in(a_in_row1),
        .acc_in(acc_00_to_10),
        .a_out(a_10_to_11),
        .acc_out(c_out_col0),
        .valid_out(val_10_to_11)
    );
    assign valid_out_col0 = val_10_to_11;

    // PE[1][1] - Bottom Right
    pe_double_buffered pe11 (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .weight_in(w11),
        .valid_in(val_10_to_11),
        .a_in(a_10_to_11),
        .acc_in(acc_01_to_11),
        .a_out(),
        .acc_out(c_out_col1),
        .valid_out(valid_out_col1)
    );

endmodule
