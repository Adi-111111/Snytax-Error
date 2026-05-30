`timescale 1ns / 1ps
module tb_mac_layer;

    // parameters matching our instance
    localparam N_INPUTS  = 4;
    localparam N_NEURONS = 4;

    // inputs
    reg         clk    = 0;
    reg         resetn = 0;
    reg         start  = 0;
    reg signed [15:0] in_val = 0;
    reg [5:0]   n_inputs = N_INPUTS;

    reg [(N_NEURONS*8)-1:0] weights_flat;

    // outputs
    wire signed [31:0] out [N_NEURONS-1:0];
    wire done;

    // clock generation
    always #5 clk = !clk;

    // instantiate the module
    mac_layer #(
        .N_INPUTS(N_INPUTS),
        .N_NEURONS(N_NEURONS),
        .APPLY_RELU(1),
        .IN_WIDTH(16),
        .ACC_WIDTH(32)
    ) uut (
        .clk(clk),
        .resetn(resetn),
        .start(start),
        .in_val(in_val),
        .weights_flat(weights_flat),
        .n_inputs(n_inputs),
        .out(out),
        .done(done)
    );

    initial begin
    $dumpfile("tb_mac.vcd");
    $dumpvars(0, tb_mac_layer);

    //pack each weight into the flat bus
    weights_flat[0  +: 8] = 8'sd16;
    weights_flat[8  +: 8] = 8'sd16;
    weights_flat[16 +: 8] = 8'sd16;
    weights_flat[24 +: 8] = 8'sd16;

    // release reset
    #20 resetn = 1;

    // TEST 1 - positive result, no ReLU clamp expected
    @(posedge clk);
    start  <= 1;
    in_val <= 16'sd16384; // 0.5
    @(posedge clk);
    start  <= 0;
    in_val <= 16'sd16384;
    @(posedge clk);
    in_val <= 16'sd16384;
    @(posedge clk);
    in_val <= 16'sd16384;

    @(posedge done);
    #1;
    $display("TEST 1 - positive inputs");
    $display("Neuron 0: %0d expected 1048576 -- %s", 
        out[0], out[0] == 1048576 ? "PASS" : "FAIL");

    // TEST 2 - negative result, ReLU should clamp to zero
    // use negative weights this time
    weights_flat[0  +: 8] = -8'sd16;
    weights_flat[8  +: 8] = -8'sd16;
    weights_flat[16 +: 8] = -8'sd16;
    weights_flat[24 +: 8] = -8'sd16;

    @(posedge clk);
    start  <= 1;
    in_val <= 16'sd16384; // 0.5 positive input, negative weight = negative result
    @(posedge clk);
    start  <= 0;
    in_val <= 16'sd16384;
    @(posedge clk);
    in_val <= 16'sd16384;
    @(posedge clk);
    in_val <= 16'sd16384;

    @(posedge done);
    #1;
    $display("TEST 2 - negative result, ReLU should clamp to zero");
    $display("Neuron 0: %0d expected 0 -- %s",
        out[0], out[0] == 0 ? "PASS" : "FAIL");

    #20 $finish;
end

    

endmodule