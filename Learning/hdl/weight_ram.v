module weight_ram #(
    parameter N_NEURONS  = 64,   // number of neurons = width of RAM
    parameter N_INPUTS   = 16    // number of inputs + 1 for bias = depth of RAM
)(
    input                           clk,

    // write port - CPU loads weights via AXI at startup
    input [15:0]                    wr_addr,
    input [7:0]                     wr_data,
    input                           wr_en,

    // read port - MAC layer fetches one full column per cycle
    input [6:0]                     rd_addr,
    output reg [(N_NEURONS*8)-1:0]  rd_data
);

    // memory array - depth is N_INPUTS+1 (extra row for biases)
    // width is N_NEURONS bytes
    reg [7:0] mem [N_NEURONS-1:0][N_INPUTS:0];

    integer j;

    always @(posedge clk) begin
        // write - CPU writes one byte at a time
        // wr_addr encodes [neuron_index, input_index]
        if (wr_en)
            mem[wr_addr[15:8]][wr_addr[7:0]] <= wr_data;

        // read - return all neuron weights for this input in one cycle
        for (j = 0; j < N_NEURONS; j = j + 1)
            rd_data[j*8 +: 8] <= mem[j][rd_addr];
    end

endmodule