// mlp_pipeline.v  — 16-cycle pipelined MLP forward pass
//
// Layer 1: 64 neurons,  N_PARALLEL=1, 16 cycles  (unchanged)
// Layer 2: 32 neurons,  N_PARALLEL=4, 16 cycles  (4 weight banks, 4 inputs/cycle)
//          128 DSP operations/cycle = 32 neurons × 4 parallel MACs
// Layer 3: 8 neurons,   N_PARALLEL=2, 16 cycles  (2 weight banks, 2 inputs/cycle)
//
// All three layers balanced at ~16 cycles each.
// n_inputs passed as full count — mac_layer divides by N_PARALLEL internally.
// Early trigger: L1(N+1) fires when l1_buf_valid pulses.

module mlp_pipeline #(
    parameter N_FEATURES = 16,
    parameter H1_SIZE    = 64,
    parameter H2_SIZE    = 32,
    parameter H3_SIZE    = 3
)(
    input                           clk,
    input                           resetn,
    input                           start,
    input signed [15:0]             x [N_FEATURES-1:0],

    // weight RAM write port (from AXI)
    // wr_bank: 0=L1, 1=L2 (all 4 banks decode internally), 2=L3
    input  [15:0]                   wr_addr,
    input  [7:0]                    wr_data,
    input                           wr_en,
    input  [1:0]                    wr_bank,

    // normalisation
    input  signed [31:0]            z_offset,
    input  [31:0]                   z_scale,

    // outputs
    output reg [7:0]                r, g, b,
    output reg                      valid,

    // expose early trigger for lookahead counter in pixel_generator
    output wire                     l1_advance_lx_out
);

    integer j;

    // -----------------------------------------------------------------------
    // Layer 1 RAM — H1_SIZE wide, N_FEATURES+1 deep, single bank
    // -----------------------------------------------------------------------
    reg  [6:0]                      l1_rd_addr;
    wire [(H1_SIZE*8)-1:0]          l1_weights;

    weight_ram #(
        .N_NEURONS(H1_SIZE),
        .N_INPUTS(N_FEATURES),
        .N_BANKS(1),
        .BANK_ID(0)
    ) l1_ram (
        .clk(clk),
        .wr_addr(wr_addr), .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd0),
        .rd_addr(l1_rd_addr),
        .rd_data(l1_weights)
    );

    // -----------------------------------------------------------------------
    // Layer 2 RAMs — H2_SIZE wide, H1_SIZE/4+1 deep, 4 banks
    // Bank k owns global columns where (col % 4) == k
    // Together deliver 4 complete weight columns per cycle
    // -----------------------------------------------------------------------
    reg  [6:0]                      l2_rd_addr;
    wire [(H2_SIZE*8)-1:0]          l2_weights_b [3:0];

    // concatenated weight bus: 4 banks × H2_SIZE neurons × 8 bits
    wire [(4*H2_SIZE*8)-1:0]        l2_weights_flat;
    assign l2_weights_flat = {l2_weights_b[3], l2_weights_b[2],
                              l2_weights_b[1], l2_weights_b[0]};

    genvar bk;
    generate
        for (bk = 0; bk < 4; bk = bk + 1) begin : gen_l2_ram
            weight_ram #(
                .N_NEURONS(H2_SIZE),
                .N_INPUTS(H1_SIZE),
                .N_BANKS(4),
                .BANK_ID(bk)
            ) l2_ram_inst (
                .clk(clk),
                .wr_addr(wr_addr), .wr_data(wr_data),
                .wr_en(wr_en && wr_bank == 2'd1),
                .rd_addr(l2_rd_addr),
                .rd_data(l2_weights_b[bk])
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Layer 3 RAMs — H3_SIZE wide, H2_SIZE/2+1 deep, 2 banks
    // -----------------------------------------------------------------------
    reg  [6:0]                      l3_rd_addr;
    wire [(H3_SIZE*8)-1:0]          l3_weights_b [1:0];
    wire [(2*H3_SIZE*8)-1:0]        l3_weights_flat;
    assign l3_weights_flat = {l3_weights_b[1], l3_weights_b[0]};

    generate
        for (bk = 0; bk < 2; bk = bk + 1) begin : gen_l3_ram
            weight_ram #(
                .N_NEURONS(H3_SIZE),
                .N_INPUTS(H2_SIZE),
                .N_BANKS(2),
                .BANK_ID(bk)
            ) l3_ram_inst (
                .clk(clk),
                .wr_addr(wr_addr), .wr_data(wr_data),
                .wr_en(wr_en && wr_bank == 2'd2),
                .rd_addr(l3_rd_addr),
                .rd_data(l3_weights_b[bk])
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Layer 1 MAC — N_PARALLEL=1, 16-bit inputs, 32-bit accumulator
    // Processes N_FEATURES=16 inputs, 1 per cycle → 16 cycles
    // -----------------------------------------------------------------------
    reg                             l1_start;
    reg  signed [15:0]              l1_in_val_flat;
    wire signed [31:0]              l1_out [H1_SIZE-1:0];
    wire                            l1_done;

    mac_layer #(
        .N_INPUTS(N_FEATURES),
        .N_NEURONS(H1_SIZE),
        .APPLY_RELU(1),
        .IN_WIDTH(16),
        .ACC_WIDTH(32),
        .N_PARALLEL(1)
    ) l1_mac (
        .clk(clk), .resetn(resetn),
        .start(l1_start),
        .in_val_flat(l1_in_val_flat),
        .weights_flat(l1_weights),
        .n_inputs(7'(N_FEATURES)),
        .out(l1_out),
        .done(l1_done)
    );

    // -----------------------------------------------------------------------
    // L1 → L2 buffer
    // -----------------------------------------------------------------------
    reg signed [31:0]               l1_buffer [H1_SIZE-1:0];
    reg                             l1_buf_valid;

    // -----------------------------------------------------------------------
    // Layer 2 MAC — N_PARALLEL=4, 32-bit inputs, 48-bit accumulator
    // 32 actual neurons, 4 inputs processed per cycle
    // n_inputs = H1_SIZE = 64, MAC takes H1_SIZE/N_PARALLEL = 16 cycles
    // DSP operations per cycle = 32 neurons × 4 parallel = 128
    // -----------------------------------------------------------------------
    reg                             l2_start;
    reg  signed [(4*32)-1:0]        l2_in_val_flat;
    wire signed [47:0]              l2_out [H2_SIZE-1:0];
    wire                            l2_done;

    mac_layer #(
        .N_INPUTS(H1_SIZE),
        .N_NEURONS(H2_SIZE),
        .APPLY_RELU(1),
        .IN_WIDTH(32),
        .ACC_WIDTH(48),
        .N_PARALLEL(4)
    ) l2_mac (
        .clk(clk), .resetn(resetn),
        .start(l2_start),
        .in_val_flat(l2_in_val_flat),
        .weights_flat(l2_weights_flat),
        .n_inputs(7'(H1_SIZE)),
        .out(l2_out),
        .done(l2_done)
    );

    // -----------------------------------------------------------------------
    // L2 → L3 buffer — sized for H2_SIZE=32 actual neurons
    // -----------------------------------------------------------------------
    reg signed [47:0]               l2_buffer [H2_SIZE-1:0];

    // -----------------------------------------------------------------------
    // Layer 3 MAC — N_PARALLEL=2, 32-bit inputs, 48-bit accumulator
    // 3 actual output neurons (RGB), 2 inputs per cycle
    // n_inputs = H2_SIZE = 32, MAC takes H2_SIZE/N_PARALLEL = 16 cycles
    // -----------------------------------------------------------------------
    reg                             l3_start;
    reg  signed [(2*32)-1:0]        l3_in_val_flat;
    wire signed [47:0]              l3_out [H3_SIZE-1:0];
    wire                            l3_done;

    mac_layer #(
        .N_INPUTS(H2_SIZE),
        .N_NEURONS(H3_SIZE),
        .APPLY_RELU(0),
        .IN_WIDTH(32),
        .ACC_WIDTH(48),
        .N_PARALLEL(2)
    ) l3_mac (
        .clk(clk), .resetn(resetn),
        .start(l3_start),
        .in_val_flat(l3_in_val_flat),
        .weights_flat(l3_weights_flat),
        .n_inputs(7'(H2_SIZE)),
        .out(l3_out),
        .done(l3_done)
    );

    // -----------------------------------------------------------------------
    // Early trigger — fire L1(N+1) the moment L1(N) finishes
    // Both L1 and L2 take 16 cycles so they stay in sync
    // -----------------------------------------------------------------------
    wire l1_early_trigger = l1_buf_valid;
    assign l1_advance_lx_out = l1_early_trigger;

    // -----------------------------------------------------------------------
    // Layer 1 control
    // -----------------------------------------------------------------------
    reg [4:0]   l1_input_idx;
    reg         l1_running;
    reg         l1_stall;

    always @(posedge clk) begin
        l1_start     <= 0;
        l1_buf_valid <= 0;
        l1_stall     <= 0;

        if (!resetn) begin
            l1_running   <= 0;
            l1_input_idx <= 0;
            l1_rd_addr   <= 0;
            l1_stall     <= 0;
        end

        else begin
            if (start || l1_early_trigger)
                l1_stall <= 1;

            if (l1_stall) begin
                l1_running   <= 1;
                l1_input_idx <= 0;
                l1_rd_addr   <= 0;
            end

            if (l1_running) begin
                l1_rd_addr     <= l1_input_idx + 1;
                l1_in_val_flat <= x[l1_input_idx];

                if (l1_input_idx == 0)
                    l1_start <= 1;

                l1_input_idx <= l1_input_idx + 1;

                if (l1_input_idx == N_FEATURES - 1)
                    l1_running <= 0;
            end

            if (l1_done) begin
                for (j = 0; j < H1_SIZE; j = j + 1)
                    l1_buffer[j] <= l1_out[j];
                l1_buf_valid <= 1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Layer 2 control — 4 inputs per cycle
    // l2_in_idx:   index into l1_buffer, steps by 4 (0,4,8,...,60)
    // l2_rd_cycle: weight bank read address, steps by 1 (0..15)
    // -----------------------------------------------------------------------
    reg         l2_running;
    reg [5:0]   l2_in_idx;
    reg [3:0]   l2_rd_cycle;

    always @(posedge clk) begin
        l2_start <= 0;

        if (!resetn) begin
            l2_running  <= 0;
            l2_in_idx   <= 0;
            l2_rd_cycle <= 0;
            l2_rd_addr  <= 0;
        end

        else begin
            if (l1_buf_valid) begin
                l2_running  <= 1;
                l2_in_idx   <= 0;
                l2_rd_cycle <= 0;
                l2_rd_addr  <= 0;
            end

            if (l2_running) begin
                l2_rd_addr <= l2_rd_cycle + 1;

                l2_in_val_flat <= {l1_buffer[l2_in_idx + 3],
                                   l1_buffer[l2_in_idx + 2],
                                   l1_buffer[l2_in_idx + 1],
                                   l1_buffer[l2_in_idx + 0]};

                if (l2_rd_cycle == 0)
                    l2_start <= 1;

                l2_in_idx   <= l2_in_idx   + 4;
                l2_rd_cycle <= l2_rd_cycle + 1;

                if (l2_rd_cycle == (H1_SIZE/4) - 1)
                    l2_running <= 0;
            end

            if (l2_done) begin
                for (j = 0; j < H2_SIZE; j = j + 1)
                    l2_buffer[j] <= l2_out[j];
            end
        end
    end

    // -----------------------------------------------------------------------
    // Layer 3 control — 2 inputs per cycle
    // l3_in_idx:   index into l2_buffer, steps by 2 (0,2,4,...,30)
    // l3_rd_cycle: weight bank read address, steps by 1 (0..15)
    // -----------------------------------------------------------------------
    reg         l3_running;
    reg [4:0]   l3_in_idx;
    reg [3:0]   l3_rd_cycle;

    always @(posedge clk) begin
        l3_start <= 0;
        valid    <= 0;

        if (!resetn) begin
            l3_running  <= 0;
            l3_in_idx   <= 0;
            l3_rd_cycle <= 0;
            l3_rd_addr  <= 0;
        end

        else begin
            if (l2_done) begin
                l3_running  <= 1;
                l3_in_idx   <= 0;
                l3_rd_cycle <= 0;
                l3_rd_addr  <= 0;
            end

            if (l3_running) begin
                l3_rd_addr <= l3_rd_cycle + 1;

                l3_in_val_flat <= {l2_buffer[l3_in_idx + 1][31:0],
                                   l2_buffer[l3_in_idx + 0][31:0]};

                if (l3_rd_cycle == 0)
                    l3_start <= 1;

                l3_in_idx   <= l3_in_idx   + 2;
                l3_rd_cycle <= l3_rd_cycle + 1;

                if (l3_rd_cycle == (H2_SIZE/2) - 1)
                    l3_running <= 0;
            end

            if (l3_done) begin
                valid <= 1;
                begin : norm
                    reg signed [63:0] r_norm, g_norm, b_norm;
                    r_norm = ((l3_out[0] - z_offset) * $signed({1'b0, z_scale})) >> 16;
                    g_norm = ((l3_out[1] - z_offset) * $signed({1'b0, z_scale})) >> 16;
                    b_norm = ((l3_out[2] - z_offset) * $signed({1'b0, z_scale})) >> 16;
                    r <= (r_norm < 0) ? 8'd0 : (r_norm > 255) ? 8'd255 : r_norm[7:0];
                    g <= (g_norm < 0) ? 8'd0 : (g_norm > 255) ? 8'd255 : g_norm[7:0];
                    b <= (b_norm < 0) ? 8'd0 : (b_norm > 255) ? 8'd255 : b_norm[7:0];
                end
            end
        end
    end

endmodule
