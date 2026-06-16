// Wide block RAM with optional multi-bank support.
// N_BANKS=1 -> single bank, full width N_NEURONS*8, depth N_INPUTS+1
// N_BANKS=2 -> odd/even column split (L3 scheme)
// N_BANKS=4 -> quarter column split (L2 scheme)
//
// Write address encoding:
//   wr_addr[15:8] = neuron index (0..N_NEURONS-1)
//   wr_addr[7:0] = global column (0..N_INPUTS)
// All banks receive the same wr_addr; only the bank where
// (global_col % N_BANKS == BANK_ID) commits the write.
//
// BRAM inference: the write path uses an explicit one-hot byte-enable vector
// decoded combinatorially from neuron_idx. This matches Vivado UG901 byte-enable
// BRAM template exactly, guaranteeing RAMB36/RAMB18 inference for all banks.

module weight_ram #(
    parameter N_NEURONS  = 64,
    parameter N_INPUTS   = 16,
    parameter N_BANKS    = 1,
    parameter BANK_ID    = 0
)(
    input                           clk,

    // write port - CPU loads weights via AXI at startup
    input  [15:0]                   wr_addr,   // [15:8]=neuron, [7:0]=global_col
    input  [7:0]                    wr_data,
    input                           wr_en,

    // read port - returns all N_NEURONS weights for one bank-local column
    input  [6:0]                    rd_addr,
    output reg [(N_NEURONS*8)-1:0]  rd_data
);

    localparam DEPTH = (N_INPUTS / N_BANKS) + 1;

    (* ram_style = "block" *)
    reg [(N_NEURONS*8)-1:0] mem [DEPTH-1:0];

    // Zero-init for simulation; BRAMs default to 0 on power-up on Zynq.
    integer init_d;
    initial begin
        for (init_d = 0; init_d < DEPTH; init_d = init_d + 1)
            mem[init_d] = {(N_NEURONS*8){1'b0}};
    end

    wire [7:0] global_col = wr_addr[7:0];
    wire [7:0] neuron_idx = wr_addr[15:8];

    wire       bank_match = ((global_col % N_BANKS) == BANK_ID[0 +: $clog2(N_BANKS+1)]);
    wire [6:0] local_row  = global_col / N_BANKS;

    // One-hot byte-enable per neuron, decoded combinatorially.
    // Vivado recognises this pattern (UG901) and infers a BRAM with per-byte WE.
    wire [N_NEURONS-1:0] byte_we;
    genvar k;
    generate
        for (k = 0; k < N_NEURONS; k = k + 1) begin : gen_bwe
            assign byte_we[k] = wr_en && bank_match && (neuron_idx == k);
        end
    endgenerate

    // Two-stage read pipeline: breaks the long BRAM output to DSP routing path.
    // Stage 1 (inside BRAM): mem[rd_addr] -> rd_data_r
    // Stage 2 (extra flop):  rd_data_r -> rd_data
    // Vivado places the extra flop near the DSPs, capping each stage at ~5ns.
    reg [(N_NEURONS*8)-1:0] rd_data_r;

    always @(posedge clk) begin
        for (integer i = 0; i < N_NEURONS; i = i + 1) begin
            if (byte_we[i])
                mem[local_row][i*8 +: 8] <= wr_data;
        end
        rd_data_r <= mem[rd_addr];  // stage 1: BRAM registered read
        rd_data   <= rd_data_r;     // stage 2: extra pipeline register
    end

endmodule
