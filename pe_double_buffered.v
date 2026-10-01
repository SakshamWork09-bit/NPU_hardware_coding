`timescale 1ns / 1ps

module pe_double_buffered (
    input  wire               clk,
    input  wire               rst_n,

    // Weight control interface
    input  wire               load_weight,   // Pulse high to load shadow register
    input  wire signed [7:0]  weight_in,     // Data bus for shadow register
    input  wire               swap_weights,  // 1-cycle strobe: shadow -> active register

    // Systolic streaming datapath
    input  wire               valid_in,
    input  wire signed [7:0]  a_in,
    input  wire signed [31:0] acc_in,

    output reg  signed [7:0]  a_out,
    output reg  signed [31:0] acc_out,
    output reg                valid_out
);

    reg signed [7:0] shadow_weight;
    reg signed [7:0] active_weight;

    always @(posedge clk) begin
        if (!rst_n) begin
            shadow_weight <= 8'sd0;
            active_weight <= 8'sd0;
            a_out         <= 8'sd0;
            acc_out       <= 32'sd0;
            valid_out     <= 1'b0;
        end else begin
            // Bank 1: Background shadow loading (independent of arithmetic)
            if (load_weight) begin
                shadow_weight <= weight_in;
            end

            // Zero-stall register bank swap
            if (swap_weights) begin
                active_weight <= shadow_weight;
            end

            // Bank 0: Active compute datapath
            if (valid_in) begin
                a_out     <= a_in;
                acc_out   <= acc_in + (a_in * active_weight);
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule
