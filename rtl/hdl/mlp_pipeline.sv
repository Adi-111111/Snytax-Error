// 16 cycle pipelined MLP forward pass
//
// Layer 1 has 64 neurons, N_PARALLEL=1, 16 cycles, 1 weight bank, 1 input/cycle
// Layer 2 has 32 neurons, N_PARALLEL=4, 16 cycles, 4 weight banks, 4 inputs/cycle
// Layer 3 has 3 neurons, N_PARALLEL=2, 16 cycles, 2 weight banks, 2 inputs/cycle
//
// DSP allocation:
// - L1: 64
// - L2: 128
// - L3: 16
// - Reserved: 12
//
// N_NEURONS reflects output neuron, N_PARALLEL reflects DSP multiplier.
// n_inputs passed at full count - MAC internally divides by N_PARALLEL.

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

    // weight RAM write port - wr_bank: 0=L1, 1=L2, 2=L3
    input  [15:0]                   wr_addr,
    input  [7:0]                    wr_data,
    input                           wr_en,
    input  [1:0]                    wr_bank,

    // normalisation inputs
    input  signed [31:0]            z_offset,
    input  [31:0]                   z_scale,
    input  [4:0]                    z_shift,

    // outputs
    output reg [7:0]                r, g, b,
    output reg                      valid,

    // early trigger output for lookahead counter in pixel_generator
    output wire                     l1_advance_lx_out
);

    integer j;

    // for normalisation:
    reg signed [63:0] r_norm, g_norm, b_norm;
    reg signed [47:0] l3_shifted  [H3_SIZE-1:0];
    reg signed [47:0] l3_diff     [H3_SIZE-1:0];
    reg signed [63:0] l3_scaled   [H3_SIZE-1:0];
    reg               l3_shift_valid;
    reg               l3_diff_valid;
    reg               l3_scaled_valid;

    // Bias constant 1 in Q0.15 used as input for bias cycles in all layers
    // Training must scale bias weights accordingly.
    localparam signed [15:0] BIAS_CONST_L1  = 16'sd32767;
    localparam signed [31:0] BIAS_CONST_L23 = 32'sd32767;

    // wr_addr[15:14] used to deterimine which layer,
    // RAMs only need [13:0]
    wire [15:0] wr_addr_masked = {2'b00, wr_addr[13:0]};

    // Layer 1 RAM - 64 neurons wide, N_FEATURES+1 deep, 1 bank
    reg  [6:0]                      l1_rd_addr;
    wire [(H1_SIZE*8)-1:0]          l1_weights;

    weight_ram #(
        .N_NEURONS(H1_SIZE), .N_INPUTS(N_FEATURES),
        .N_BANKS(1), .BANK_ID(0)
    ) l1_ram (
        .clk(clk),
        .wr_addr(wr_addr_masked), .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd0),
        .rd_addr(l1_rd_addr), .rd_data(l1_weights)
    );

    // Layer 2 RAMs - 32 neurons wide, (H1_SIZE/4)+1 deep, 4 banks
    // Bank k owns global columns where (col % 4) == k
    // Each bank delivers 32x8=256 bits per cycle
    // 4 banks / parallel inputs x H2_SIZE neurons x 8 bits all concatenated into 1024-bit flattned bus
    reg  [6:0]                      l2_rd_addr;
    wire [(H2_SIZE*8)-1:0]          l2_weights_b [3:0];

    wire [(4*H2_SIZE*8)-1:0]        l2_weights_flat;
    assign l2_weights_flat = {l2_weights_b[3], l2_weights_b[2],
                              l2_weights_b[1], l2_weights_b[0]};

    genvar bk;
    generate
        for (bk = 0; bk < 4; bk = bk + 1) begin : gen_l2_ram
            weight_ram #(
                .N_NEURONS(H2_SIZE), .N_INPUTS(H1_SIZE),
                .N_BANKS(4), .BANK_ID(bk)
            ) l2_ram_inst (
                .clk(clk),
                .wr_addr(wr_addr_masked), .wr_data(wr_data),
                .wr_en(wr_en && wr_bank == 2'd1),
                .rd_addr(l2_rd_addr),
                .rd_data(l2_weights_b[bk])
            );
        end
    endgenerate

    // Layer 3 RAMs - 8 neurons wide, (H2_SIZE/2)+1 deep, 2 banks
    reg  [6:0]                      l3_rd_addr;
    wire [(H3_SIZE*8)-1:0]          l3_weights_b [1:0];
    wire [(2*H3_SIZE*8)-1:0]        l3_weights_flat;
    assign l3_weights_flat = {l3_weights_b[1], l3_weights_b[0]};

    generate
        for (bk = 0; bk < 2; bk = bk + 1) begin : gen_l3_ram
            weight_ram #(
                .N_NEURONS(H3_SIZE), .N_INPUTS(H2_SIZE),
                .N_BANKS(2), .BANK_ID(bk)
            ) l3_ram_inst (
                .clk(clk),
                .wr_addr(wr_addr_masked), .wr_data(wr_data),
                .wr_en(wr_en && wr_bank == 2'd2),
                .rd_addr(l3_rd_addr),
                .rd_data(l3_weights_b[bk])
            );
        end
    endgenerate

    // Layer 1 MAC - 64 neurons, N_PARALLEL=1, 16-bit inputs, 32-bit accumulator
    // Takes N_FEATURES = 16 cycles here
    reg                             l1_start;
    reg  signed [15:0]              l1_in_val_flat;
    wire signed [47:0]              l1_out [H1_SIZE-1:0];
    wire                            l1_done;

    mac_layer #(
        .N_INPUTS(N_FEATURES), .N_NEURONS(H1_SIZE), .APPLY_RELU(1),
        .IN_WIDTH(16), .ACC_WIDTH(48), .N_PARALLEL(1)
    ) l1_mac (
        .clk(clk), .resetn(resetn),
        .start(l1_start),
        .in_val_flat(l1_in_val_flat),
        .weights_flat(l1_weights),
        .n_inputs(7'(N_FEATURES + 1)),  // +1 for bias cycle
        .out(l1_out), .done(l1_done)
    );

    // L1 -> L2 buffer
    reg signed [31:0]               l1_buffer [H1_SIZE-1:0];
    reg                             l1_buf_valid;

    // Layer 2 MAC - 32 neurons, N_PARALLEL=4, 32-bit inputs, 48-bit accumulator
    // n_inputs = H1_SIZE = 64 (MAC internally takes 64/4 = 16 cycles)
    // 32 neurons x 4 parallel inputs = 128 DSP operations per cycle
    reg                             l2_start;
    reg  signed [(4*32)-1:0]        l2_in_val_flat;
    wire signed [47:0]              l2_out [H2_SIZE-1:0];  // 32 actual outputs
    wire                            l2_done;

    mac_layer #(
        .N_INPUTS(H1_SIZE),
        .N_NEURONS(H2_SIZE),
        .APPLY_RELU(1),
        .IN_WIDTH(32), .ACC_WIDTH(48), .N_PARALLEL(4)
    ) l2_mac (
        .clk(clk), .resetn(resetn),
        .start(l2_start),
        .in_val_flat(l2_in_val_flat),
        .weights_flat(l2_weights_flat),
        .n_inputs(7'(H1_SIZE + 1)),   // +1 for bias cycle (same 17 cycles total)
        .out(l2_out), .done(l2_done)
    );

    // L2 -> L3 buffer - sized to H2_SIZE = 32 actual neurons here
    reg signed [47:0]               l2_buffer [H2_SIZE-1:0];

    // Layer 3 MAC - 3 neurons (H3_SIZE), N_PARALLEL=2, 32-bit inputs, 48-bit accumulator
    // n_inputs = H2_SIZE = 32 (MAC internally takes 32/2 = 16 cycles)
    reg                             l3_start;
    reg  signed [(2*32)-1:0]        l3_in_val_flat;
    wire signed [47:0]              l3_out [H3_SIZE-1:0];
    wire                            l3_done;

    mac_layer #(
        .N_INPUTS(H2_SIZE),
        .N_NEURONS(H3_SIZE),
        .APPLY_RELU(0),
        .IN_WIDTH(32), .ACC_WIDTH(48), .N_PARALLEL(2)
    ) l3_mac (
        .clk(clk), .resetn(resetn),
        .start(l3_start),
        .in_val_flat(l3_in_val_flat),
        .weights_flat(l3_weights_flat),
        .n_inputs(7'(H2_SIZE + 1)),   // +1 for bias cycle (17 cycles also)
        .out(l3_out), .done(l3_done)
    );

    // Early trigger: L1(N+1) fires the moment l1_buf_valid pulses.
    // L1 takes 16 cycles. L2 takes H1_SIZE/N_PARALLEL = 16 cycles.
    // With 1-cycle stall they complete together.
    wire l1_early_trigger = l1_buf_valid;
    assign l1_advance_lx_out = l1_early_trigger;

    // Layer 1 control
    reg [4:0]   l1_input_idx;
    reg         l1_running;
    reg         l1_stall;
    reg         l1_stall2;  // extra cycle for 2-stage BRAM read

    always @(posedge clk) begin
        l1_start     <= 0;
        l1_buf_valid <= 0;
        l1_stall     <= 0;
        l1_stall2    <= 0;

        if (!resetn) begin
            l1_running   <= 0;
            l1_input_idx <= 0;
            l1_rd_addr   <= 0;
            l1_stall     <= 0;
            l1_stall2    <= 0;
        end

        else begin
            if (start || l1_early_trigger)
                l1_stall <= 1;

            // stall: present addr 0 to BRAM, queue stall2
            if (l1_stall) begin
                l1_stall2    <= 1;
                l1_input_idx <= 0;
                l1_rd_addr   <= 0;
            end

            // stall2: present addr 1 to BRAM, then start running
            if (l1_stall2) begin
                l1_running <= 1;
                l1_rd_addr <= 1;
            end

            if (l1_running) begin
                l1_rd_addr     <= l1_input_idx + 2;  // 2-cycle prefetch for 2-stage BRAM

                // bias cycle: inject constant 1.0 instead of feature input
                if (l1_input_idx == N_FEATURES)
                    l1_in_val_flat <= BIAS_CONST_L1;
                else
                    l1_in_val_flat <= x[l1_input_idx];

                if (l1_input_idx == 0)
                    l1_start <= 1;

                l1_input_idx <= l1_input_idx + 1;

                if (l1_input_idx == N_FEATURES)  // one extra cycle for bias
                    l1_running <= 0;
            end

            if (l1_done) begin
                for (j = 0; j < H1_SIZE; j = j + 1)
                    l1_buffer[j] <= l1_out[j][31:0];  // safe: ReLU bounds value to ~27 bits
                l1_buf_valid <= 1;
            end
        end
    end

    // Layer 2 control - 4 inputs per cycle
    // l2_in_idx: index into l1_buffer, steps by 4 (0,4,8,...,60)
    // l2_rd_cycle: weight bank read address, steps by 1 (0..15)
    reg         l2_running;
    reg         l2_init;    // extra cycle for 2-stage BRAM read
    reg [5:0]   l2_in_idx;
    reg [4:0]   l2_rd_cycle;  // needs to count to H1_SIZE/4 = 16

    always @(posedge clk) begin
        l2_start <= 0;
        l2_init  <= 0;

        if (!resetn) begin
            l2_running  <= 0;
            l2_init     <= 0;
            l2_in_idx   <= 0;
            l2_rd_cycle <= 0;
            l2_rd_addr  <= 0;
        end

        else begin
            // l1_buf_valid: present addr 0, queue init cycle
            if (l1_buf_valid) begin
                l2_init     <= 1;
                l2_in_idx   <= 0;
                l2_rd_cycle <= 0;
                l2_rd_addr  <= 0;
            end

            // init: present addr 1, then start running
            if (l2_init) begin
                l2_running <= 1;
                l2_rd_addr <= 1;
            end

            if (l2_running) begin
                l2_rd_addr <= l2_rd_cycle + 2;  // 2-cycle prefetch

                // bias cycle: only uses lane 0 for BIAS_CONST, lanes 1-3 = 0
                // bias weights live in bank 0 row H1_SIZE/4; banks 1-3 add 0
                if (l2_rd_cycle == H1_SIZE/4)
                    l2_in_val_flat <= {96'h0, BIAS_CONST_L23};
                else begin
                    l2_in_val_flat <= {l1_buffer[l2_in_idx + 3],
                                       l1_buffer[l2_in_idx + 2],
                                       l1_buffer[l2_in_idx + 1],
                                       l1_buffer[l2_in_idx + 0]};
                    l2_in_idx <= l2_in_idx + 4;
                end

                if (l2_rd_cycle == 0)
                    l2_start <= 1;

                l2_rd_cycle <= l2_rd_cycle + 1;

                if (l2_rd_cycle == H1_SIZE/4)  // one extra cycle for bias
                    l2_running <= 0;
            end

            if (l2_done) begin
                // capture all H2_SIZE=32 actual neuron outputs
                for (j = 0; j < H2_SIZE; j = j + 1)
                    l2_buffer[j] <= l2_out[j];
            end
        end
    end

    // Layer 3 control - 2 inputs per cycle
    // l3_in_idx: index into l2_buffer, steps by 2 (0,2,4,...,30)
    // l3_rd_cycle: weight bank read address, steps by 1 (0..15)
    reg         l3_running;
    reg         l3_init;    // extra cycle for 2-stage BRAM read
    reg [4:0]   l3_in_idx;
    reg [4:0]   l3_rd_cycle;  // needs to count to H2_SIZE/2 = 16

    always @(posedge clk) begin
        l3_start        <= 0;
        valid           <= 0;
        l3_init         <= 0;
        l3_shift_valid  <= 0;
        l3_diff_valid   <= 0;
        l3_scaled_valid <= 0;

        if (!resetn) begin
            l3_running      <= 0;
            l3_init         <= 0;
            l3_in_idx       <= 0;
            l3_rd_cycle     <= 0;
            l3_rd_addr      <= 0;
            l3_shift_valid  <= 0;
            l3_diff_valid   <= 0;
            l3_scaled_valid <= 0;
        end

        else begin
            // l2_done: present addr 0, queue init cycle
            if (l2_done) begin
                l3_init     <= 1;
                l3_in_idx   <= 0;
                l3_rd_cycle <= 0;
                l3_rd_addr  <= 0;
            end

            // init: present addr 1, then start running
            if (l3_init) begin
                l3_running <= 1;
                l3_rd_addr <= 1;
            end

            if (l3_running) begin
                l3_rd_addr <= l3_rd_cycle + 2;  // 2-cycle prefetch

                // bias cycle: only uses lane 0 for BIAS_CONST, lane 1 = 0
                // bias weights live in bank 0 row H2_SIZE/2; bank 1 adds 0
                if (l3_rd_cycle == H2_SIZE/2)
                    l3_in_val_flat <= {32'h0, BIAS_CONST_L23};
                else begin
                    l3_in_val_flat <= {l2_buffer[l3_in_idx + 1][31:0],
                                       l2_buffer[l3_in_idx + 0][31:0]};
                    l3_in_idx <= l3_in_idx + 2;
                end

                if (l3_rd_cycle == 0)
                    l3_start <= 1;

                l3_rd_cycle <= l3_rd_cycle + 1;

                if (l3_rd_cycle == H2_SIZE/2)  // one extra cycle for bias
                    l3_running <= 0;
            end

            // Output normalisation pipeline has 4 stages:

            // Stage 1: arithmetic right shift by z_shift (collapses +-500B into +- hundreds range)
            if (l3_done) begin
                for (j = 0; j < H3_SIZE; j = j + 1)
                    l3_shifted[j] <= $signed(l3_out[j]) >>> z_shift;
                l3_shift_valid <= 1;
            end

            // Stage 2: subtract z_offset (to fit into 32-bit signed)
            if (l3_shift_valid) begin
                for (j = 0; j < H3_SIZE; j = j + 1)
                    l3_diff[j] <= l3_shifted[j] - z_offset;
                l3_diff_valid <= 1;
            end

            // Stage 3: multiply z_scale
            if (l3_diff_valid) begin
                for (j = 0; j < H3_SIZE; j = j + 1)
                    // using $signed to keep negative l3_diff correct
                    l3_scaled[j] <= $signed(l3_diff[j]) * $signed({1'b0, z_scale});
                l3_scaled_valid <= 1;
            end

            // Stage 4: shift >>>16 and clamp to 8-bit RGB
            if (l3_scaled_valid) begin
                valid  <= 1;
                // arithmetic shift preserves sign
                r_norm = l3_scaled[0] >>> 16;
                g_norm = l3_scaled[1] >>> 16;
                b_norm = l3_scaled[2] >>> 16;
                r <= (r_norm < 0) ? 8'd0 : (r_norm > 255) ? 8'd255 : r_norm[7:0];
                g <= (g_norm < 0) ? 8'd0 : (g_norm > 255) ? 8'd255 : g_norm[7:0];
                b <= (b_norm < 0) ? 8'd0 : (b_norm > 255) ? 8'd255 : b_norm[7:0];
            end
        end
    end

endmodule
