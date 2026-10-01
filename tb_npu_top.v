`timescale 1ns / 1ps

module tb_npu_top;

    reg               clk;
    reg               rst_n;
    reg               start;`r`n    time start_time;
    reg  signed [7:0]  w00, w01, w10, w11;
    reg  signed [7:0]  a_col0_row0, a_col0_row1;
    reg  signed [7:0]  a_col1_row0, a_col1_row1;

    wire signed [7:0]  out_col0_int8, out_col1_int8;
    wire              out_valid_col0, out_valid_col1;
    wire              busy, done;

    npu_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .a_col0_row0(a_col0_row0), .a_col0_row1(a_col0_row1),
        .a_col1_row0(a_col1_row0), .a_col1_row1(a_col1_row1),
        .out_col0_int8(out_col0_int8),
        .out_col1_int8(out_col1_int8),
        .out_valid_col0(out_valid_col0),
        .out_valid_col1(out_valid_col1),
        .busy(busy),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("npu_top.vcd");
        $dumpvars(0, tb_npu_top);

        clk   = 0;
        rst_n = 0;
        start = 0;

        // Weights: [[2, 3], [4, 5]]
        w00 = 8'sd2; w01 = 8'sd3;
        w10 = 8'sd4; w11 = 8'sd5;

        // Inputs: [[1, 2], [3, 4]]
        // Expected Raw C: [[10, 13], [22, 29]]
        // Expected Scaled (>>> 1): [[5, 6], [11, 14]]
        a_col0_row0 = 8'sd1; a_col0_row1 = 8'sd3;
        a_col1_row0 = 8'sd2; a_col1_row1 = 8'sd4;

        #20;
        @(negedge clk);
        rst_n = 1;

        @(negedge clk);`r`n        start = 1;`r`n        start_time = $time;`r`n        $display("START: %0t", start_time);`r`n`r`n        @(negedge clk);`r`n        start = 0;`r`n`r`n        @(posedge done);`r`n        $display("DONE: %0t", $time);`r`n        $display("NPU LATENCY: %0t", $time - start_time);
        #30;
        $finish;
    end

endmodule



