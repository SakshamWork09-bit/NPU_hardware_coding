`timescale 1ns / 1ps

module tb_mac_int8;

    reg signed [7:0] a_in;
    reg signed [7:0] b_in;
    reg              clk;
    reg              rst_n;
    reg              valid_in;
    wire             valid_out;
    wire signed [31:0] accum_out;

    // Instantiate Unit Under Test
    mac_int8 uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .a_in(a_in),
        .b_in(b_in),
        .valid_out(valid_out),
        .accum_out(accum_out)
    );

    // 100 MHz reference simulation clock (period = 10ns)
    always #5 clk = ~clk;

    initial begin
        $dumpfile("mac_int8.vcd");
        $dumpvars(0, tb_mac_int8);

        // Initialize all inputs
        clk      = 0;
        rst_n    = 0;
        valid_in = 0;
        a_in     = 8'sd0;
        b_in     = 8'sd0;

        // Hold reset active for 2 full cycles
        #20;
        @(negedge clk);
        rst_n = 1;

        // Cycle 1: Drive inputs on negedge -> MAC samples on posedge
        // Expected math: 10 * 5 = 50 -> Accum = 50
        @(negedge clk);
        valid_in = 1;
        a_in     = 8'sd10;
        b_in     = 8'sd5;

        // Cycle 2: (-12) * 4 = -48 -> Accum = 50 + (-48) = 2
        @(negedge clk);
        a_in     = -8'sd12;
        b_in     = 8'sd4;

        // Cycle 3: 7 * 3 = 21 -> Accum = 2 + 21 = 23
        @(negedge clk);
        a_in     = 8'sd7;
        b_in     = 8'sd3;

        // De-assert valid; accumulator should latch and hold 23
        @(negedge clk);
        valid_in = 0;
        a_in     = 8'sd0;
        b_in     = 8'sd0;

        // Allow pipeline to settle and check result
        #20;
        if (accum_out === 32'sd23) begin
            $display("[TEST PASSED] Final Accumulator Value: %0d", accum_out);
        end else begin
            $display("[TEST FAILED] Expected 23, Got: %0d", accum_out);
        end

        $finish;
    end

endmodule
