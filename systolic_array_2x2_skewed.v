`timescale 1ns / 1ps

module systolic_array_2x2_skewed (
    input  wire               clk,
    input  wire               rst_n,

    // Double-buffered weight interface
    input  wire               load_weight,
    input  wire               swap_weights,
    input  wire signed [7:0]  w00,
    input  wire signed [7:0]  w01,
    input  wire signed [7:0]  w10,
    input  wire signed [7:0]  w11,

    // Natural, UN-SKEWED matrix input bus (fed simultaneously from memory)
    input  wire               matrix_valid_in,
    input  wire signed [7:0]  row0_data_in,
    input  wire signed [7:0]  row1_data_in,

    // Matrix outputs (South boundary)
    output wire signed [31:0] c_out_col0,
    output wire signed [31:0] c_out_col1,
    output wire               valid_out_col0,
    output wire               valid_out_col1
);

    // -------------------------------------------------------------
    // Hardware Skew Buffers (Row 0 = wire, Row 1 = 1-cycle pipeline)
    // -------------------------------------------------------------
    wire signed [7:0] a_row0_skewed;
    wire              valid_row0_skewed;
    reg  signed [7:0] a_row1_skewed;
    reg               valid_row1_skewed;

    // Row 0 enters immediately
    assign a_row0_skewed     = row0_data_in;
    assign valid_row0_skewed = matrix_valid_in;

    // Row 1 passes through physical DFF buffer to delay by 1 clock cycle
    always @(posedge clk) begin
        if (!rst_n) begin
            a_row1_skewed     <= 8'sd0;
            valid_row1_skewed <= 1'b0;
        end else begin
            a_row1_skewed     <= row1_data_in;
            valid_row1_skewed <= matrix_valid_in;
        end
    end

    // -------------------------------------------------------------
    // Core 2x2 Double-Buffered Systolic Grid
    // -------------------------------------------------------------
    systolic_array_2x2_db array_core (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .valid_in_row0(valid_row0_skewed),
        .valid_in_row1(valid_row1_skewed),
        .a_in_row0(a_row0_skewed),
        .a_in_row1(a_row1_skewed),
        .c_out_col0(c_out_col0),
        .c_out_col1(c_out_col1),
        .valid_out_col0(valid_out_col0),
        .valid_out_col1(valid_out_col1)
    );

endmodule
