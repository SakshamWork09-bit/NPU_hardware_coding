`timescale 1ns / 1ps

module tb_npu_core_2x2;

    reg               clk;
    reg               rst_n;
    reg               start;
    reg  signed [7:0]  w00, w01, w10, w11;
    reg  signed [7:0]  a_col0_row0, a_col0_row1;
    reg  signed [7:0]  a_col1_row0, a_col1_row1;

    wire signed [31:0] c_out_col0, c_out_col1;
    wire              valid_out_col0, valid_out_col1;
    wire              busy, done;

    npu_core_2x2 uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .w00(w00), .w01(w01), .w10(w10), .w11(w11),
        .a_col0_row0(a_col0_row0), .a_col0_row1(a_col0_row1),
        .a_col1_row0(a_col1_row0), .a_col1_row1(a_col1_row1),
        .c_out_col0(c_out_col0), .c_out_col1(c_out_col1),
        .valid_out_col0(valid_out_col0), .valid_out_col1(valid_out_col1),
        .busy(busy), .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("npu_core_2x2.vcd");
        $dumpvars(0, tb_npu_core_2x2);

        clk   = 0;
        rst_n = 0;
        start = 0;

        // Weights: [[2, 3], [4, 5]]
        w00 = 8'sd2; w01 = 8'sd3;
        w10 = 8'sd4; w11 = 8'sd5;

        // Activation Matrix A: [[1, 2], [3, 4]]
        a_col0_row0 = 8'sd1; a_col0_row1 = 8'sd3;
        a_col1_row0 = 8'sd2; a_col1_row1 = 8'sd4;

        #20;
        @(negedge clk);
        rst_n = 1;

        // Trigger autonomous hardware execution with a single clock pulse
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        // Wait until hardware signals completion via interrupt
        @(posedge done);
        #20;
        $finish;
    end

endmodule
