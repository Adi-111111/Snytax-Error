`timescale 1ns / 1ps
// Tests the z_shift normalization register added to mlp_pipeline.
//
// All weights are valid Q4.4 values (int8 = round(float * 16)):
//   x[0..3]    = 16384  (0.5 in Q1.15)
//   L1/L2 wts  = 16     (1.0 in Q4.4)
//   L3 R wt    = 64     (4.0 in Q4.4)
//   L3 G wt    = 32     (2.0 in Q4.4)
//   L3 B wt    = -32    (-2.0 in Q4.4)
//   all biases = 0
//
// Trace:
//   l1_acc = 4 * 16384 * 16 = 1048576      -> l1_buf = 1048576
//   l2_acc = 4 * 1048576 * 16 = 67108864   -> l2_buf = 67108864
//   l3_acc = [4*67108864*64, 4*67108864*32, 4*67108864*(-32)]
//           = [2^34, 2^33, -2^33]
//           = [17179869184, 8589934592, -8589934592]
//
// Five tests sweep z_shift, z_offset, z_scale and check clamped RGB output.

module tb_zshift;

    localparam N_FEATURES = 4;
    localparam H1_SIZE    = 4;
    localparam H2_SIZE    = 4;
    localparam H3_SIZE    = 3;

    reg clk    = 0;
    reg resetn = 0;
    reg start  = 0;

    reg signed [15:0] x [N_FEATURES-1:0];

    reg [15:0] wr_addr = 0;
    reg [7:0]  wr_data = 0;
    reg        wr_en   = 0;
    reg [1:0]  wr_bank = 0;

    reg signed [31:0] z_offset = 0;
    reg [31:0]        z_scale  = 65536;
    reg [4:0]         z_shift  = 0;

    wire [7:0] r, g, b;
    wire valid;

    always #4 clk = ~clk;  // 125 MHz

    mlp_pipeline #(
        .N_FEATURES(N_FEATURES),
        .H1_SIZE   (H1_SIZE),
        .H2_SIZE   (H2_SIZE),
        .H3_SIZE   (H3_SIZE)
    ) dut (
        .clk     (clk),
        .resetn  (resetn),
        .start   (start),
        .x       (x),
        .wr_addr (wr_addr),
        .wr_data (wr_data),
        .wr_en   (wr_en),
        .wr_bank (wr_bank),
        .z_offset(z_offset),
        .z_scale (z_scale),
        .z_shift (z_shift),
        .r(r), .g(g), .b(b),
        .valid   (valid),
        .l1_advance_lx_out()
    );

    // -----------------------------------------------------------------------
    // Write one weight byte into the RAM
    // -----------------------------------------------------------------------
    task write_weight;
        input [1:0] bank;
        input [7:0] neuron;
        input [7:0] col;
        input [7:0] data;
        begin
            @(posedge clk);
            wr_bank <= bank;
            wr_addr <= (neuron << 8) | col;
            wr_data <= data;
            wr_en   <= 1;
            @(posedge clk);
            wr_en   <= 0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Reset the pipeline, load norm params, fire one pixel, check output
    // -----------------------------------------------------------------------
    task run_test;
        input [4:0]         shift;
        input signed [31:0] offset;
        input [31:0]        scale;
        input [7:0]         exp_r, exp_g, exp_b;
        input integer       test_num;
        begin
            // Short reset clears pipeline valid signals (RAM weights are preserved)
            resetn = 0; repeat(4) @(posedge clk);
            z_shift  = shift;
            z_offset = offset;
            z_scale  = scale;
            resetn = 1; repeat(4) @(posedge clk);

            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            @(posedge valid); #1;

            if (r == exp_r && g == exp_g && b == exp_b)
                $display("TEST %0d PASS  z_shift=%2d  z_offset=%4d  z_scale=%5d  ->  R=%3d G=%3d B=%3d",
                         test_num, shift, $signed(offset), scale, r, g, b);
            else
                $display("TEST %0d FAIL  z_shift=%2d  z_offset=%4d  z_scale=%5d  ->  R=%3d G=%3d B=%3d  (expected R=%3d G=%3d B=%3d)",
                         test_num, shift, $signed(offset), scale, r, g, b, exp_r, exp_g, exp_b);
        end
    endtask

    integer i, j;

    initial begin
        $dumpfile("tb_zshift.vcd");
        $dumpvars(0, tb_zshift);

        // All inputs = 0.5 in Q1.15
        for (i = 0; i < N_FEATURES; i = i + 1)
            x[i] = 16384;

        resetn = 1; repeat(4) @(posedge clk);

        // L1: all weights = 16 (1.0 in Q4.4), bias = 0
        for (j = 0; j < H1_SIZE; j = j + 1) begin
            for (i = 0; i < N_FEATURES; i = i + 1)
                write_weight(2'd0, j[7:0], i[7:0], 8'd16);
            write_weight(2'd0, j[7:0], N_FEATURES[7:0], 8'd0);
        end

        // L2: all weights = 16 (1.0 in Q4.4), bias = 0
        for (j = 0; j < H2_SIZE; j = j + 1) begin
            for (i = 0; i < H1_SIZE; i = i + 1)
                write_weight(2'd1, j[7:0], i[7:0], 8'd16);
            write_weight(2'd1, j[7:0], H1_SIZE[7:0], 8'd0);
        end

        // L3 neuron 0 (R): weight = +64, bias = 0  (bank 2 = L3 per wr_bank comment)
        for (i = 0; i < H2_SIZE; i = i + 1)
            write_weight(2'd2, 8'd0, i[7:0], 8'd64);
        write_weight(2'd2, 8'd0, H2_SIZE[7:0], 8'd0);

        // L3 neuron 1 (G): weight = +32, bias = 0
        for (i = 0; i < H2_SIZE; i = i + 1)
            write_weight(2'd2, 8'd1, i[7:0], 8'd32);
        write_weight(2'd2, 8'd1, H2_SIZE[7:0], 8'd0);

        // L3 neuron 2 (B): weight = -32 (0xE0 two's-complement), bias = 0
        for (i = 0; i < H2_SIZE; i = i + 1)
            write_weight(2'd2, 8'd2, i[7:0], 8'hE0);
        write_weight(2'd2, 8'd2, H2_SIZE[7:0], 8'd0);

        // -------------------------------------------------------------------
        // l3_acc = [2^34, 2^33, -2^33] = [17179869184, 8589934592, -8589934592]
        // Q4.4 weights (int8=16 = 1.0) give 2x larger accumulator than int8=8,
        // so z_shift is 2 higher than the old int8=8 tests.
        // -------------------------------------------------------------------

        // TEST 1: z_shift=26, z_offset=0, z_scale=65536
        //   shifted = [256, 128, -128]   (R overflows by 1 — clamps to 255)
        //   diff    = [256, 128, -128]
        //   >>16    = [256, 128, -128]   -> clamp -> [255, 128, 0]
        run_test(5'd26, 32'sd0, 32'd65536, 8'd255, 8'd128, 8'd0, 1);

        // TEST 2: z_shift=27, z_offset=0, z_scale=65536
        //   shifted = [128, 64, -64]
        //   diff    = [128, 64, -64]
        //   >>16    = [128, 64, -64]   -> clamp -> [128, 64, 0]
        run_test(5'd27, 32'sd0, 32'd65536, 8'd128, 8'd64, 8'd0, 2);

        // TEST 3: z_shift=27, z_offset=-64, z_scale=65536
        //   shifted = [128, 64, -64]
        //   diff    = [192, 128,   0]   (subtract −64 adds 64 to each)
        //   >>16    = [192, 128,   0]   -> no clamp needed
        run_test(5'd27, -32'sd64, 32'd65536, 8'd192, 8'd128, 8'd0, 3);

        // TEST 4: z_shift=0, z_offset=0, z_scale=65536
        //   no shift — giant values saturate the clamp -> [255, 255, 0]
        run_test(5'd0, 32'sd0, 32'd65536, 8'd255, 8'd255, 8'd0, 4);

        // TEST 5: z_shift=26, z_offset=0, z_scale=32768
        //   shifted = [256, 128, -128]
        //   *32768>>16 = [128, 64, -64]   -> clamp -> [128, 64, 0]
        run_test(5'd26, 32'sd0, 32'd32768, 8'd128, 8'd64, 8'd0, 5);

        #100;
        $display("Done.");
        $finish;
    end

endmodule
