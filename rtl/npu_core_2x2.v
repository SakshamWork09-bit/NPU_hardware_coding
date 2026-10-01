`timescale 1ns / 1ps

module npu_core_2x2 (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    
    // Weight Matrix Bus (Configured before start)
    input  wire signed [7:0]  w00, w01, w10, w11,
    
    // Input Matrix Memory Buffer Interface
    input  wire signed [7:0]  a_col0_row0, a_col0_row1, // Col 0 of A
    input  wire signed [7:0]  a_col1_row0, a_col1_row1, // Col 1 of A

    // Array Outputs
    output wire signed [31:0] c_out_col0,
    output wire signed [31:0] c_out_col1,
    output wire               valid_out_col0,
    output wire               valid_out_col1,
    output wire               busy,
    output wire               done
);

    wire       load_weight;
    wire       swap_weights;
    wire       matrix_valid_in;
    wire [1:0] stream_idx;

    // Multiplexer selecting matrix column based on FSM stream counter
    wire signed [7:0] row0_data = (stream_idx == 2'd0) ? a_col0_row0 : a_col1_row0;
    wire signed [7:0] row1_data = (stream_idx == 2'd0) ? a_col0_row1 : a_col1_row1;

    // Instantiate FSM Autopilot
    systolic_controller ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .matrix_valid_in(matrix_valid_in),
        .stream_idx(stream_idx),
        .busy(busy),
        .done(done)
    );

    // Instantiate Hardware Array with Skew Buffers
    systolic_array_2x2_skewed array_subsystem (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .matrix_valid_in(matrix_valid_in),
        .row0_data_in(row0_data),
        .row1_data_in(row1_data),
        .c_out_col0(c_out_col0),
        .c_out_col1(c_out_col1),
        .valid_out_col0(valid_out_col0),
        .valid_out_col1(valid_out_col1)
    );

endmodule
