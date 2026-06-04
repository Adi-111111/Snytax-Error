// weight_ram.v
// Wide block RAM with optional multi-bank support.
// N_BANKS=1  -> single bank, full width N_NEURONS*8, depth N_INPUTS+1
// N_BANKS=2  -> odd/even column split (original L2 scheme)
// N_BANKS=4  -> quarter column split (new 4-wide L2 scheme)
//
// All N_BANKS instances share the same module; BANK_ID selects which
// columns this instance owns. Column k belongs to bank (k % N_BANKS).
// Read address is a simple cycle counter 0..N_INPUTS/N_BANKS.
// Write address encoding:
//   wr_addr[15:8] = neuron index  (0..N_NEURONS-1)
//   wr_addr[7:0]  = column index  (0..N_INPUTS)   <- global column, not bank-local
// The RAM internally checks (column % N_BANKS == BANK_ID) before writing,
// so the CPU can issue the same wr_addr to all banks; only the matching
// bank commits the write.

module weight_ram #(
    parameter N_NEURONS  = 64,      // number of neurons = word width in bytes
    parameter N_INPUTS   = 16,      // number of inputs (excl. bias)
    parameter N_BANKS    = 1,       // 1, 2, or 4
    parameter BANK_ID    = 0        // which bank this instance is (0..N_BANKS-1)
)(
    input                           clk,

    // write port — CPU loads weights via AXI at startup
    // all banks receive the same signals; only the matching bank commits
    input  [15:0]                   wr_addr,   // [15:8]=neuron, [7:0]=global_col
    input  [7:0]                    wr_data,
    input                           wr_en,

    // read port — one read per cycle, returns all N_NEURONS weights for
    // the bank-local column addressed by rd_addr
    input  [6:0]                    rd_addr,   // bank-local column index
    output reg [(N_NEURONS*8)-1:0]  rd_data
);

    // local depth: each bank owns every N_BANKS-th column
    // +1 for the bias row
    localparam DEPTH_ACTUAL = (N_INPUTS / N_BANKS) + 1;
    localparam DEPTH = (DEPTH_ACTUAL < 512) ? 512 : DEPTH_ACTUAL;

    // memory: rows = DEPTH, columns = N_NEURONS bytes
    (* ram_style = "block" *) reg [(N_NEURONS*8)-1:0] mem [DEPTH-1:0];

    // initialize to 0 so unwritten rows (e.g. bias row) behave like FPGA BRAM
    integer init_d;
    initial begin
        for (init_d = 0; init_d < DEPTH; init_d = init_d + 1)
            mem[init_d] = 0;
    end
 

    // write — decode global column to bank-local row
    wire [7:0] global_col  = wr_addr[7:0];
    wire [7:0] neuron_idx  = wr_addr[15:8];

    // this bank owns global columns where (col % N_BANKS) == BANK_ID

    wire bank_match = ((global_col % N_BANKS) == BANK_ID);
    wire [6:0] local_row   = global_col / N_BANKS;  // bank-local row address

    // write - was: mem[neuron_idx][local_row] <= wr_data;
    always @(posedge clk) begin
        if (wr_en && bank_match && neuron_idx < N_NEURONS)
            mem[local_row][neuron_idx*8 +: 8] <= wr_data;

        // read - was: rd_data[j*8 +: 8] <= mem[j][rd_addr];
        rd_data <= mem[rd_addr];
    end
 

endmodule