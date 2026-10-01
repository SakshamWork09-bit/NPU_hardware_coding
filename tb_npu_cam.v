`timescale 1ns / 1ps

module tb_npu_cam;

    reg               clk;
    reg               rst_n;
    reg               start;

    reg signed [7:0]  w00, w01, w10, w11;
    reg signed [7:0]  a_col0_row0, a_col0_row1;
    reg signed [7:0]  a_col1_row0, a_col1_row1;

    wire signed [7:0] out_col0_int8;
    wire signed [7:0] out_col1_int8;
    wire              out_valid_col0;
    wire              out_valid_col1;
    wire              busy;
    wire              done;

    // Instantiate Top-Level NPU with verified port names
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

    // 100 MHz clock
    always #5 clk = ~clk;

    reg signed [7:0] pixel_mem [0:4095];
    integer out_file, in_file;
    integer num_tiles;
    integer t_idx;
    integer status;

    // Registers to capture 2x2 tile outputs deterministically
    reg [1:0] col0_cnt;
    reg [1:0] col1_cnt;
    reg signed [7:0] tile_c00, tile_c10, tile_c01, tile_c11;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col0_cnt <= 0;
            col1_cnt <= 0;
            tile_c00 <= 0; tile_c10 <= 0;
            tile_c01 <= 0; tile_c11 <= 0;
        end else begin
            if (start) begin
                col0_cnt <= 0;
                col1_cnt <= 0;
            end else begin
                if (out_valid_col0) begin
                    if (col0_cnt == 0) tile_c00 <= out_col0_int8;
                    else               tile_c10 <= out_col0_int8;
                    col0_cnt <= col0_cnt + 1;
                end
                if (out_valid_col1) begin
                    if (col1_cnt == 0) tile_c01 <= out_col1_int8;
                    else               tile_c11 <= out_col1_int8;
                    col1_cnt <= col1_cnt + 1;
                end
            end
        end
    end

    initial begin
        clk   = 0;
        rst_n = 0;
        start = 0;
        a_col0_row0 = 0; a_col0_row1 = 0;
        a_col1_row0 = 0; a_col1_row1 = 0;

        // Spatial Edge Kernel: [[2, -2], [-2, 2]]
        w00 = 8'sd2;   w01 = -8'sd2;
        w10 = -8'sd2;  w11 = 8'sd2;

        $readmemh("cam_in.hex", pixel_mem);

        out_file = $fopen("cam_out.hex", "w");
        if (out_file == 0) begin
            $display("ERROR: Cannot open cam_out.hex for writing.");
            $finish;
        end

        in_file = $fopen("tile_count.txt", "r");
        if (in_file != 0) begin
            status = $fscanf(in_file, "%d\n", num_tiles);
            $fclose(in_file);
        end else begin
            num_tiles = 256;
        end

        #20 rst_n = 1;
        #20;

        for (t_idx = 0; t_idx < num_tiles; t_idx = t_idx + 1) begin
            @(posedge clk);
            a_col0_row0 = pixel_mem[t_idx*4 + 0];
            a_col0_row1 = pixel_mem[t_idx*4 + 2];
            a_col1_row0 = pixel_mem[t_idx*4 + 1];
            a_col1_row1 = pixel_mem[t_idx*4 + 3];

            start = 1;
            @(posedge clk);
            start = 0;
            @(posedge clk);

            while (!done) begin
                @(posedge clk);
            end

            // Write 2x2 tile outputs in raster order: C00, C01, C10, C11
            $fwrite(out_file, "%02x %02x %02x %02x\n",
                    tile_c00 & 8'hFF,
                    tile_c01 & 8'hFF,
                    tile_c10 & 8'hFF,
                    tile_c11 & 8'hFF);

            @(posedge clk);
        end

        #50;
        $fclose(out_file);
        $finish;
    end

endmodule
