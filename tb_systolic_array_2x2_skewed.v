`timescale 1ns / 1ps

module tb_systolic_array_2x2_skewed;

    reg               clk;
    reg               rst_n;
    reg               load_weight;
    reg               swap_weights;
    reg  signed [7:0]  w00, w01, w10, w11;
    reg               matrix_valid_in;
    reg  signed [7:0]  row0_data_in;
    reg  signed [7:0]  row1_data_in;

    wire signed [31:0] c_out_col0, c_out_col1;
    wire              valid_out_col0, valid_out_col1;

    systolic_array_2x2_skewed uut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .matrix_valid_in(matrix_valid_in),
        .row0_data_in(row0_data_in),
        .row1_data_in(row1_data_in),
        .c_out_col0(c_out_col0),
        .c_out_col1(c_out_col1),
        .valid_out_col0(valid_out_col0),
        .valid_out_col1(valid_out_col1)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("systolic_skewed.vcd");
        $dumpvars(0, tb_systolic_array_2x2_skewed);

        clk             = 0;
        rst_n           = 0;
        load_weight     = 0;
        swap_weights    = 0;
        matrix_valid_in = 0;
        row0_data_in    = 0;
        row1_data_in    = 0;
        w00 = 0; w01 = 0; w10 = 0; w11 = 0;

        #20;
        @(negedge clk);
        rst_n = 1;

        // --- Load Weights: W = [[2, 3], [4, 5]] ---
        @(negedge clk);
        load_weight = 1;
        w00 = 8'sd2; w01 = 8'sd3;
        w10 = 8'sd4; w11 = 8'sd5;

        @(negedge clk);
        load_weight  = 0;
        swap_weights = 1;

        @(negedge clk);
        swap_weights = 0;

        // --- Stream Matrix A in UN-SKEWED natural format ---
        // Matrix A:
        //   [1, 2]  <- Row 0
        //   [3, 4]  <- Row 1
        // Expected C = A x W = [[10, 13], [22, 29]]

        // Cycle 1: Stream Column 0 of Matrix A (Row0=1, Row1=3) AT THE SAME TIME
        @(negedge clk);
        matrix_valid_in = 1;
        row0_data_in    = 8'sd1;
        row1_data_in    = 8'sd3;

        // Cycle 2: Stream Column 1 of Matrix A (Row0=2, Row1=4) AT THE SAME TIME
        @(negedge clk);
        row0_data_in    = 8'sd2;
        row1_data_in    = 8'sd4;

        // Cycle 3: Stream complete, return to idle
        @(negedge clk);
        matrix_valid_in = 0;
        row0_data_in    = 8'sd0;
        row1_data_in    = 8'sd0;

        #50;
        $finish;
    end

endmodule
