`timescale 1ns / 1ps
module tb_input_builder;

    localparam N_FEATURES = 4;

    reg         clk    = 0;
    reg         resetn = 0;
    reg [9:0]   pixel_x = 0;
    reg [8:0]   pixel_y = 0;
    reg [5:0]   axis_x_select = 0;
    reg [5:0]   axis_y_select = 1;

    reg signed [15:0] feature_regs [N_FEATURES-1:0];
    wire signed [15:0] x [N_FEATURES-1:0];

    always #5 clk = !clk;

    input_builder #(
        .N_FEATURES(N_FEATURES)
    ) uut (
        .clk(clk),
        .resetn(resetn),
        .pixel_x(pixel_x),
        .pixel_y(pixel_y),
        .axis_x_select(axis_x_select),
        .axis_y_select(axis_y_select),
        .feature_regs(feature_regs),
        .x(x)
    );

    initial begin
        $dumpfile("tb_input.vcd");
        $dumpvars(0, tb_input_builder);

        // set feature registers
        feature_regs[0] = 16'sd8192;  // 0.25
        feature_regs[1] = 16'sd16384; // 0.5
        feature_regs[2] = 16'sd24576; // 0.75
        feature_regs[3] = 16'sd32767; // ~1.0

        #20 resetn = 1;

        // x select = 0, y select = 1

        // TEST 1 - middle of screen
        // Expected values (Q1.15)
        // x[0] should be pixel_x/639 = 320/639 ~ 0.5
        // x[1] should be pixel_y/479 = 240/479 ~ 0.5
        // x[2] should be feature_regs[2] ~ 0.75
        // x[3] should be feature_regs[3] ~ 1.0
        pixel_x = 320;
        pixel_y = 240;
        @(posedge clk);
        @(posedge clk); // one cycle latency
        #1;
        $display("TEST 1 - middle of screen");
        // inequalities here used to account for fixed point approximation
        $display("x[0] = %0d expected ~16320 (0.498) -- %s", x[0], (x[0] > 16000 && x[0] < 16500) ? "PASS" : "FAIL");
        $display("x[1] = %0d expected ~16320 (0.498) -- %s", x[1], (x[1] > 16000 && x[1] < 16500) ? "PASS" : "FAIL");
        $display("x[2] = %0d expected 24576 (0.75)  -- %s", x[2], x[2] == 24576 ? "PASS" : "FAIL");
        $display("x[3] = %0d expected 32767 (~1.0)  -- %s", x[3], x[3] == 32767 ? "PASS" : "FAIL");

        // TEST 2 - top left corner
        // Expected values
        // x[0] should be 0
        // x[1] should be 0
        pixel_x = 0;
        pixel_y = 0;
        @(posedge clk);
        @(posedge clk);
        #1;
        $display("TEST 2 - top left corner");
        $display("x[0] = %0d expected 0 -- %s", x[0], x[0] == 0 ? "PASS" : "FAIL");
        $display("x[1] = %0d expected 0 -- %s", x[1], x[1] == 0 ? "PASS" : "FAIL");

        // TEST 3 - bottom right corner
        // Expected values (Q1.15)
        // x[0] should be ~ 1
        // x[1] should be ~ 1
        pixel_x = 639;
        pixel_y = 479;
        @(posedge clk);
        @(posedge clk);
        #1;
        $display("TEST 3 - bottom right corner");
        $display("x[0] = %0d expected ~32589 (~1.0) -- %s", x[0], (x[0] > 32000) ? "PASS" : "FAIL");
        $display("x[1] = %0d expected ~32572 (~1.0) -- %s", x[1], (x[1] > 32000) ? "PASS" : "FAIL");

        #20 $finish;
    end

endmodule