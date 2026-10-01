`timescale 1ns / 1ps

module tb_systolic_array_2x2_db;

    reg               clk;
    reg               rst_n;
    reg               load_weight;
    reg               swap_weights;
    reg  signed [7:0]  w00, w01, w10, w11;
    reg               valid_in_row0, valid_in_row1;
    reg  signed [7:0]  a_in_row0, a_in_row1;

    wire signed [31:0] c_out_col0, c_out_col1;
    wire              valid_out_col0, valid_out_col1;

    systolic_array_2x2_db uut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .swap_weights(swap_weights),
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

    always #5 clk = ~clk;

    initial begin
        $dumpfile("systolic_2x2_db.vcd");
        $dumpvars(0, tb_systolic_array_2x2_db);

        clk           = 0;
        rst_n         = 0;
        load_weight   = 0;
        swap_weights  = 0;
        w00 = 0; w01 = 0; w10 = 0; w11 = 0;
        valid_in_row0 = 0; valid_in_row1 = 0;
        a_in_row0     = 0; a_in_row1     = 0;

        #20;
        @(negedge clk);
        rst_n = 1;

        // --- PHASE 1: Load Layer 1 Weights into Shadow Registers ---
        // W1 = [[2, 3], [4, 5]]
        @(negedge clk);
        load_weight = 1;
        w00 = 8'sd2; w01 = 8'sd3;
        w10 = 8'sd4; w11 = 8'sd5;

        // Instant swap to active banks
        @(negedge clk);
        load_weight  = 0;
        swap_weights = 1;

        // --- PHASE 2: Compute Matrix 1 WHILE pre-loading Layer 2 Weights in Background ---
        // Layer 1 Activations: A = [[1, 2], [3, 4]] -> Expected C1 = [[10, 13], [22, 29]]
        // Layer 2 Target Weights: W2 = [[1, -1], [2, 1]]
        
        // Cycle 1: Stream A[0][0]=1 on Row0. Trigger shadow load of W2 simultaneously!
        @(negedge clk);
        swap_weights  = 0;
        valid_in_row0 = 1;
        a_in_row0     = 8'sd1;
        load_weight   = 1; // Background write begins
        w00 = 8'sd1; w01 = -8'sd1;
        w10 = 8'sd2; w11 =  8'sd1;

        // Cycle 2: Stream A[0][1]=2 on Row0, skewed A[1][0]=3 on Row1.
        @(negedge clk);
        load_weight   = 0; // Shadow load finished. Shadow registers hold W2!
        a_in_row0     = 8'sd2;
        valid_in_row1 = 1;
        a_in_row1     = 8'sd3;

        // Cycle 3: Stream A[1][1]=4 on Row1.
        @(negedge clk);
        a_in_row0     = 8'sd0;
        valid_in_row0 = 0;
        a_in_row1     = 8'sd4;

        // --- PHASE 3: Zero-Stall Layer Transition (Swap & Stream Layer 2 Instantly) ---
        // Layer 2 Activations: B = [[2, 1], [1, 3]] -> Expected C2 = [[4, -1], [7, 2]]
        
        // Cycle 4: Assert swap_weights right as Layer 2 Row0 begins!
        @(negedge clk);
        swap_weights  = 1;
        valid_in_row0 = 1;
        a_in_row0     = 8'sd2; // B[0][0]
        a_in_row1     = 8'sd0;
        valid_in_row1 = 0;

        // Cycle 5: De-assert swap, continue streaming B
        @(negedge clk);
        swap_weights  = 0;
        a_in_row0     = 8'sd1; // B[0][1]
        valid_in_row1 = 1;
        a_in_row1     = 8'sd1; // B[1][0] (skewed)

        // Cycle 6: Stream B[1][1]
        @(negedge clk);
        a_in_row0     = 8'sd0;
        valid_in_row0 = 0;
        a_in_row1     = 8'sd3; // B[1][1]

        // Cycle 7: Flush pipeline
        @(negedge clk);
        valid_in_row1 = 0;
        a_in_row1     = 8'sd0;

        #50;
        $finish;
    end

endmodule
