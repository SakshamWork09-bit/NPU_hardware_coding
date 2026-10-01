`timescale 1ns / 1ps

module mac_int8 (
    input  wire               clk,
    input  wire               rst_n,      // Active-low synchronous reset
    input  wire               valid_in,   // High when input data is valid
    input  wire signed [7:0]  a_in,
    input  wire signed [7:0]  b_in,
    output reg                valid_out,  // High when output holds valid math
    output reg  signed [31:0] accum_out
);

    // Non-blocking assignments (<=) infer physical D-Flip-Flops clocked on posedge clk
    always @(posedge clk) begin
        if (!rst_n) begin
            accum_out <= 32'sd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            accum_out <= accum_out + (a_in * b_in);
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
