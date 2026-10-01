`timescale 1ns / 1ps

module tb_pe_double_buffered;

    reg               clk;
    reg               rst_n;
    reg               load_weight;
    reg  signed [7:0]  weight_in;
    reg               swap_weights;
    reg               valid_in;
    reg  signed [7:0]  a_in;
    reg  signed [31:0] acc_in;

    wire signed [7:0]  a_out;
    wire signed [31:0] acc_out;
    wire              valid_out;

    pe_double_buffered uut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .weight_in(weight_in),
        .swap_weights(swap_weights),
        .valid_in(valid_in),
        .a_in(a_in),
        .acc_in(acc_in),
        .a_out(a_out),
        .acc_out(acc_out),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("pe_double_buffered.vcd");
        $dumpvars(0, tb_pe_double_buffered);

        clk          = 0;
        rst_n        = 0;
        load_weight  = 0;
        weight_in    = 8'sd0;
        swap_weights = 0;
        valid_in     = 0;
        a_in         = 8'sd0;
        acc_in       = 32'sd0;

        #20;
        @(negedge clk);
        rst_n = 1;

        // Phase 1: Initialize Layer 1 (W = 5)
        @(negedge clk);
        load_weight = 1;
        weight_in   = 8'sd5;

        @(negedge clk);
        load_weight  = 0;
        swap_weights = 1;

        @(negedge clk);
        swap_weights = 0;

        // Phase 2: Compute Layer 1 WHILE loading Layer 2 (W = -3) in background
        @(negedge clk);
        valid_in    = 1;
        a_in        = 8'sd4;
        acc_in      = 32'sd10;  // 10 + (4 * 5) = 30
        load_weight = 1;
        weight_in   = -8'sd3;

        @(negedge clk);
        a_in        = 8'sd6;
        acc_in      = 32'sd0;   // 0 + (6 * 5) = 30
        load_weight = 0;

        // Phase 3: Strobe swap at boundary
        @(negedge clk);
        swap_weights = 1;
        a_in         = 8'sd2;
        acc_in       = 32'sd10;  // 10 + (2 * 5) = 20 (Final token of Layer 1)

        // Phase 4: First token of Layer 2 (Active weight is now -3)
        @(negedge clk);
        swap_weights = 0;
        a_in         = 8'sd7;
        acc_in       = 32'sd100; // 100 + (7 * -3) = 79

        @(negedge clk);
        valid_in = 0;
        a_in     = 8'sd0;
        acc_in   = 32'sd0;

        #30;
        $finish;
    end

endmodule
