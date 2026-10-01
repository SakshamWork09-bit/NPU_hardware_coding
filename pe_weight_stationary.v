`timescale 1ns / 1ps

module pe_weight_stationary (
    input  wire               clk,
    input  wire               rst_n,
    
    // Control interface
    input  wire               load_weight, // Asserted to latch weight into local register
    input  wire signed [7:0]  weight_in,
    input  wire               valid_in,    // High when valid activation & accumulator arrive
    
    // Systolic data paths
    input  wire signed [7:0]  a_in,        // From West neighbor
    input  wire signed [31:0] acc_in,      // From North neighbor
    
    output reg  signed [7:0]  a_out,       // Registered pass-through to East neighbor
    output reg  signed [31:0] acc_out,     // Registered MAC result to South neighbor
    output reg                valid_out    // Downstream valid indicator
);

    // Stationary weight storage register
    reg signed [7:0] weight_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg <= 8'sd0;
            a_out      <= 8'sd0;
            acc_out    <= 32'sd0;
            valid_out  <= 1'b0;
        end else begin
            // Weight latching stage
            if (load_weight) begin
                weight_reg <= weight_in;
            end

            // Spatial execution stage
            if (valid_in) begin
                a_out     <= a_in;                           // Forward activation East
                acc_out   <= acc_in + (a_in * weight_reg);   // Accumulate South
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule
