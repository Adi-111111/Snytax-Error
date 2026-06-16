`timescale 1ns / 1ps
module tb_weight_ram;

    localparam N_NEURONS = 4;
    localparam N_INPUTS  = 4;

    reg         clk    = 0;
    reg [15:0]  wr_addr = 0;
    reg [7:0]   wr_data = 0;
    reg         wr_en   = 0;
    reg [5:0]   rd_addr = 0;

    wire [(N_NEURONS*8)-1:0] rd_data;

    always #5 clk = !clk;

    weight_ram #(
        .N_NEURONS(N_NEURONS),
        .N_INPUTS(N_INPUTS)
    ) uut (
        .clk(clk),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_en(wr_en),
        .rd_addr(rd_addr),
        .rd_data(rd_data)
    );

    // weight writing task for simplicity
    task write_weight;
        input [7:0] neuron;
        input [7:0] input_idx;
        input [7:0] data;
        begin
            @(posedge clk);
            wr_addr <= (neuron << 8) | input_idx;
            wr_data <= data;
            wr_en   <= 1;
            @(posedge clk);
            wr_en   <= 0;
        end
    endtask

    initial begin
        $dumpfile("tb_ram.vcd");
        $dumpvars(0, tb_weight_ram);

        // TEST 1 - write known weights and read them back
        // write weight 1.0 (= 16 in Q4.4) for all neurons at input 0
        #20;
        write_weight(0, 0, 8'sd16);  // neuron 0, input 0 = 1.0
        write_weight(1, 0, 8'sd32);  // neuron 1, input 0 = 2.0
        write_weight(2, 0, 8'sd8);   // neuron 2, input 0 = 0.5
        write_weight(3, 0, -8'sd16); // neuron 3, input 0 = -1.0

        // read back column 0 - should get all four weights simultaneously
        @(posedge clk);
        rd_addr <= 0;
        @(posedge clk); // one cycle latency
        @(posedge clk); // wait for rd_data to settle
        #1;

        $display("TEST 1 - read column 0");
        $display("Neuron 0 weight: %0d expected 16  -- %s", $signed(rd_data[0 +: 8]), $signed(rd_data[0 +: 8]) == 16 ? "PASS" : "FAIL");
        $display("Neuron 1 weight: %0d expected 32  -- %s", $signed(rd_data[8 +: 8]), $signed(rd_data[8 +: 8]) == 32 ? "PASS" : "FAIL");
        $display("Neuron 2 weight: %0d expected 8   -- %s", $signed(rd_data[16 +: 8]), $signed(rd_data[16 +: 8]) == 8 ? "PASS" : "FAIL");
        $display("Neuron 3 weight: %0d expected -16 -- %s", $signed(rd_data[24 +: 8]), $signed(rd_data[24 +: 8]) == -16 ? "PASS" : "FAIL");

        // TEST 2 - write and read biases (at column N_INPUTS)
        write_weight(0, N_INPUTS, 8'sd4);  // neuron 0 bias = 0.25
        write_weight(1, N_INPUTS, 8'sd8);  // neuron 1 bias = 0.5

        @(posedge clk);
        rd_addr <= N_INPUTS;
        @(posedge clk);
        @(posedge clk);
        #1;

        $display("TEST 2 - read bias column");
        $display("Neuron 0 bias: %0d expected 4 -- %s", $signed(rd_data[0 +: 8]), $signed(rd_data[0 +: 8]) == 4 ? "PASS" : "FAIL");
        $display("Neuron 1 bias: %0d expected 8 -- %s", $signed(rd_data[8 +: 8]), $signed(rd_data[8 +: 8]) == 8 ? "PASS" : "FAIL");

        #20 $finish;
    end

endmodule
