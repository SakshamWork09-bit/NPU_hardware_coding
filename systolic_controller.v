`timescale 1ns / 1ps

module systolic_controller (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,          // Host command: begin matrix multiplication

    // Control signals driven to the systolic array
    output reg        load_weight,
    output reg        swap_weights,
    output reg        matrix_valid_in,
    output reg  [1:0] stream_idx,     // Selects which column of A to feed
    output reg        busy,
    output reg        done            // 1-cycle interrupt back to host
);

    localparam [2:0] S_IDLE   = 3'b000,
                     S_LOAD_W = 3'b001,
                     S_SWAP_W = 3'b010,
                     S_STREAM = 3'b011,
                     S_DRAIN  = 3'b100,
                     S_DONE   = 3'b101;

    reg [2:0] state, next_state;
    reg [2:0] drain_cnt;

    // Sequential State Register
    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            stream_idx <= 2'd0;
            drain_cnt  <= 3'd0;
        end else begin
            state <= next_state;

            if (state == S_STREAM)
                stream_idx <= stream_idx + 1'b1;
            else
                stream_idx <= 2'd0;

            if (state == S_DRAIN)
                drain_cnt <= drain_cnt + 1'b1;
            else
                drain_cnt <= 3'd0;
        end
    end

    // Next-State Combinational Logic
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:   if (start) next_state = S_LOAD_W;
            S_LOAD_W: next_state = S_SWAP_W;
            S_SWAP_W: next_state = S_STREAM;
            S_STREAM: if (stream_idx == 2'd1) next_state = S_DRAIN; // 2x2 matrix -> 2 streaming cycles
            S_DRAIN:  if (drain_cnt == 3'd3)  next_state = S_DONE;   // Wait 4 cycles for array drain
            S_DONE:   next_state = S_IDLE;
            default:  next_state = S_IDLE;
        endcase
    end

    // Autonomous Control Output Logic
    always @(*) begin
        load_weight     = 1'b0;
        swap_weights    = 1'b0;
        matrix_valid_in = 1'b0;
        busy            = 1'b1;
        done            = 1'b0;

        case (state)
            S_IDLE: begin
                busy = 1'b0;
            end
            S_LOAD_W: begin
                load_weight = 1'b1;
            end
            S_SWAP_W: begin
                swap_weights = 1'b1;
            end
            S_STREAM: begin
                matrix_valid_in = 1'b1;
            end
            S_DRAIN: begin
                matrix_valid_in = 1'b0;
            end
            S_DONE: begin
                done = 1'b1;
                busy = 1'b0;
            end
        endcase
    end

endmodule
