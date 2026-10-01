`timescale 1ns / 1ps

module tb_pe;

    reg               clk;
    reg               rst_n;
    reg               load_weight;
    reg  signed [7:0]  weight_in;
    reg               valid_in;
    reg  signed [7:0]  a_in;
    reg  signed [31:0] acc_in;

    wire signed [7:0]  a_out;
    wire signed [31:0] acc_out;
    wire              valid_out;

    // Instantiate Unit Under Test
    pe_weight_stationary uut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .weight_in(weight_in),
        .valid_in(valid_in),
        .a_in(a_in),
        .acc_in(acc_in),
        .a_out(a_out),
        .acc_out(acc_out),
        .valid_out(valid_out)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        $dumpfile("pe_sim.vcd");
        $dumpvars(0, tb_pe);

        // Initialize signals
        clk         = 0;
        rst_n       = 0;
        load_weight = 0;
        weight_in   = 8'sd0;
        valid_in    = 0;
        a_in        = 8'sd0;
        acc_in      = 32'sd0;

        // Reset across 2 clock periods
        #20;
        @(negedge clk);
        rst_n = 1;

        // Phase 1: Pre-load stationary weight W = 6
        @(negedge clk);
        load_weight = 1;
        weight_in   = 8'sd6;

        @(negedge clk);
        load_weight = 0;
        weight_in   = 8'sd0;

        // Phase 2: Stream Token 1 -> a_in = 3, North partial sum acc_in = 10
        // Math: acc_out = 10 + (3 * 6) = 28; a_out = 3
        @(negedge clk);
        valid_in = 1;
        a_in     = 8'sd3;
        acc_in   = 32'sd10;

        // Phase 3: Stream Token 2 -> a_in = -4, North partial sum acc_in = 50
        // Math: acc_out = 50 + (-4 * 6) = 26; a_out = -4
        @(negedge clk);
        a_in     = -8'sd4;
        acc_in   = 32'sd50;

        // Phase 4: Stream Token 3 -> a_in = 12, North partial sum acc_in = -20
        // Math: acc_out = -20 + (12 * 6) = 52; a_out = 12
        @(negedge clk);
        a_in     = 8'sd12;
        acc_in   = -32'sd20;

        // Deassert valid
        @(negedge clk);
        valid_in = 0;
        a_in     = 8'sd0;
        acc_in   = 32'sd0;

        #30;
        $finish;
    end

endmodule
