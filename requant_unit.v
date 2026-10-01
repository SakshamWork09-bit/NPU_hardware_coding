`timescale 1ns / 1ps

module requant_unit #(
    parameter SHIFT_BITS = 1 // Dynamic scaling factor
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               valid_in,
    input  wire signed [31:0] acc_in,
    output reg                valid_out,
    output reg  signed [7:0]  data_out
);

    wire signed [31:0] relu_val;
    wire signed [31:0] scaled_val;

    // 1. ReLU: Non-linear activation
    assign relu_val = (acc_in > 32'sd0) ? acc_in : 32'sd0;

    // 2. Fixed-point arithmetic scaling
    assign scaled_val = relu_val >>> SHIFT_BITS;

    // 3. Synchronous saturating clamp to signed INT8 [0 to 127]
    always @(posedge clk) begin
        if (!rst_n) begin
            data_out  <= 8'sd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                if (scaled_val > 32'sd127)
                    data_out <= 8'sd127;
                else
                    data_out <= scaled_val[7:0];
            end else begin
                data_out <= 8'sd0;
            end
        end
    end

endmodule
