`timescale 1ns / 1ps

module tb_systolic_array_2x2;

    reg               clk;
    reg               rst_n;
    reg               load_weight;
    reg  signed [7:0]  w00, w01, w10, w11;
    reg               valid_in_row0;
    reg               valid_in_row1;
    reg  signed [7:0]  a_in_row0;
    reg  signed [7:0]  a_in_row1;

    wire signed [31:0] c_out_col0;
    wire signed [31:0] c_out_col1;
    wire              valid_out_col0;
    wire              valid_out_col1;

    systolic_array_2x2 uut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .valid_in_row0(valid_in_row0),
        .valid_in_row1(valid_in_row1),
        .a_in_row0(a_in_row0),
        .a_in_row1(a_in_row1),
        .c_out_col0(c_out_col0),
        .c_out_col1(c_out_col1),
        .valid_out_col0(valid_out_col0),
        .valid_out_col1(valid_out_col1)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        $dumpfile("systolic_2x2.vcd");
        $dumpvars(0, tb_systolic_array_2x2);

        clk           = 0;
        rst_n         = 0;
        load_weight   = 0;
        w00 = 8'sd0; w01 = 8'sd0; w10 = 8'sd0; w11 = 8'sd0;
        valid_in_row0 = 0; valid_in_row1 = 0;
        a_in_row0     = 8'sd0; a_in_row1 = 8'sd0;

        #20;
        @(negedge clk);
        rst_n = 1;

        // Step 1: Pre-load Weights W = [[2, 3], [4, 5]]
        @(negedge clk);
        load_weight = 1;
        w00 = 8'sd2; w01 = 8'sd3;
        w10 = 8'sd4; w11 = 8'sd5;

        @(negedge clk);
        load_weight = 0;

        // Step 2: Stream Matrix A = [[1, 2], [3, 4]] with 1-cycle Row Skew
        // Time T1: Row 0 gets A00 (1). Row 1 waits (skew cycle).
        @(negedge clk);
        valid_in_row0 = 1; a_in_row0 = 8'sd1;
        valid_in_row1 = 0; a_in_row1 = 8'sd0;

        // Time T2: Row 0 gets A10 (3). Row 1 gets A01 (2).
        @(negedge clk);
        valid_in_row0 = 1; a_in_row0 = 8'sd3;
        valid_in_row1 = 1; a_in_row1 = 8'sd2;

        // Time T3: Row 0 is done. Row 1 gets A11 (4).
        @(negedge clk);
        valid_in_row0 = 0; a_in_row0 = 8'sd0;
        valid_in_row1 = 1; a_in_row1 = 8'sd4;

        // Time T4: Feed complete. Pipeline drains.
        @(negedge clk);
        valid_in_row1 = 0; a_in_row1 = 8'sd0;

        #60;
        $finish;
    end

endmodule
