`timescale 1ns / 1ps
module tb_mlp_pipeline;

    localparam N_FEATURES = 4;
    localparam H1_SIZE    = 4;
    localparam H2_SIZE    = 4;
    localparam H3_SIZE    = 3;

    reg         clk    = 0;
    reg         resetn = 0;
    reg         start  = 0;

    reg signed [15:0] x [N_FEATURES-1:0];

    reg [15:0]  wr_addr = 0;
    reg [7:0]   wr_data = 0;
    reg         wr_en   = 0;
    reg [1:0]   wr_bank = 0;

    reg signed [31:0] z_offset = 0;
    reg signed [31:0] z_scale  = 32'h00010000; // 1.0 in Q16.16

    wire [7:0]  r, g, b;
    wire        valid;

    always #4 clk = !clk; // 125MHz

    mlp_pipeline #(
        .N_FEATURES(N_FEATURES),
        .H1_SIZE(H1_SIZE),
        .H2_SIZE(H2_SIZE),
        .H3_SIZE(H3_SIZE)
    ) uut (
        .clk(clk),
        .resetn(resetn),
        .start(start),
        .x(x),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_en(wr_en),
        .wr_bank(wr_bank),
        .z_offset(z_offset),
        .z_scale(z_scale),
        .r(r), .g(g), .b(b),
        .valid(valid)
    );

    // task to write one weight byte
    task write_weight;
        input [1:0]  bank;
        input [7:0]  neuron;
        input [7:0]  input_idx;
        input [7:0]  data;
        begin
            @(posedge clk);
            wr_bank <= bank;
            wr_addr <= (neuron << 8) | input_idx;
            wr_data <= data;
            wr_en   <= 1;
            @(posedge clk);
            wr_en   <= 0;
        end
    endtask

    integer i, j;

    initial begin
        $dumpfile("tb_mlp.vcd");
        $dumpvars(0, tb_mlp_pipeline);

        // set input vector - all 0.5 in Q1.15 = 16384
        for (i = 0; i < N_FEATURES; i = i + 1)
            x[i] = 16'sd16384;

        // release reset
        #20 resetn = 1;
        #20;

        // load layer 1 weights - all 0.5 in Q4.4 = 8
        // neuron j, input i = 8
        for (j = 0; j < H1_SIZE; j = j + 1)
            for (i = 0; i < N_FEATURES; i = i + 1)
                write_weight(0, j, i, 8'sd8);

        // load layer 1 biases - all zero
        for (j = 0; j < H1_SIZE; j = j + 1)
            write_weight(0, j, N_FEATURES, 8'sd0);

        // load layer 2 weights - all 0.5 in Q4.4 = 8
        for (j = 0; j < H2_SIZE; j = j + 1)
            for (i = 0; i < H1_SIZE; i = i + 1)
                write_weight(1, j, i, 8'sd8); // even bank

        // load layer 2 biases - all zero
        for (j = 0; j < H2_SIZE; j = j + 1)
            write_weight(1, j, H1_SIZE, 8'sd0);

        // load layer 3 weights - all 0.5 in Q4.4 = 8
        for (j = 0; j < H3_SIZE; j = j + 1)
            for (i = 0; i < H2_SIZE; i = i + 1)
                write_weight(3, j, i, 8'sd8);

        // load layer 3 biases - all zero
        for (j = 0; j < H3_SIZE; j = j + 1)
            write_weight(3, j, H2_SIZE, 8'sd0);

        #20;

        // send one pixel
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        // wait for valid output
        @(posedge valid);
        #1;
        $display("RGB output: R=%0d G=%0d B=%0d", r, g, b);
        $display("Valid high - pixel complete");

        #100 $finish;
    end

endmodule