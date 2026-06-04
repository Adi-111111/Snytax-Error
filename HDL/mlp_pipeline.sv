module mlp_pipeline #(
    parameter int N_FEATURES = 16,
    parameter int H1_SIZE    = 64,
    parameter int H2_SIZE    = 32,
    parameter int H3_SIZE    = 3
)(
    input  logic        clk,
    input  logic        resetn,
    input  logic        start,
    input  logic signed [15:0] x [N_FEATURES-1:0],

    input  logic [15:0] wr_addr,
    input  logic [7:0]  wr_data,
    input  logic        wr_en,
    input  logic [1:0]  wr_bank,

    input  logic signed [31:0] z_offset [2:0],
    input  logic        [31:0] z_scale  [2:0],

    output logic [7:0]  r, g, b,
    output logic        valid,
    output logic        l1_advance_lx_out
);

    // -------------------------------------------------------
    // Layer 1 BRAMs - 64 neurons split lo/hi (32 neurons each)
    // Write port: 8-bit data, 10-bit address
    // Read port:  256-bit data, 5-bit address
    // -------------------------------------------------------
    logic [6:0]             l1_rd_addr;
    logic [(H1_SIZE*8)-1:0] l1_weights;
    logic [(H1_SIZE*8)-1:0] l1_weights_r;

    wire l1_we    = wr_en && (wr_bank == 2'd0);
    wire l1_lo_we = l1_we && (wr_addr[15:8] < 8'd32);
    wire l1_hi_we = l1_we && (wr_addr[15:8] >= 8'd32);

    wire [9:0] l1_lo_waddr   = {wr_addr[12:8], wr_addr[4:0]};
    wire [9:0] l1_hi_waddr   = {(wr_addr[15:8] - 8'd32), wr_addr[4:0]};
    wire [4:0] l1_bram_raddr = l1_rd_addr[4:0];

    wire [255:0] l1_rd_lo, l1_rd_hi;

    l1_weights_bram_lo l1_bram_lo (
        .clka  (clk),             .ena   (1'b1),
        .wea   ({1{l1_lo_we}}),   .addra (l1_lo_waddr),  .dina (wr_data),
        .clkb  (clk),
        .addrb (l1_bram_raddr),   .doutb (l1_rd_lo)
    );

    l1_weights_bram_hi l1_bram_hi (
        .clka  (clk),             .ena   (1'b1),
        .wea   ({1{l1_hi_we}}),   .addra (l1_hi_waddr),  .dina (wr_data),
        .clkb  (clk),
        .addrb (l1_bram_raddr),   .doutb (l1_rd_hi)
    );

    assign l1_weights = {l1_rd_hi, l1_rd_lo};

    // Register BRAM output to break the RAMB36 output -> DSP critical path.
    // Control logic sends rd_addr 2 cycles ahead to compensate.
    always_ff @(posedge clk)
        l1_weights_r <= l1_weights;

    // -------------------------------------------------------
    // Layer 2 BRAM - 32 neurons
    // Write port: 8-bit data, 12-bit address
    // Read port:  256-bit data, 7-bit address
    // -------------------------------------------------------
    logic [6:0]             l2_rd_addr;
    logic [(H2_SIZE*8)-1:0] l2_weights;
    logic [(H2_SIZE*8)-1:0] l2_weights_r;

    wire l2_we = wr_en && (wr_bank == 2'd1);
    wire [11:0] l2_waddr = {wr_addr[12:8], wr_addr[6:0]};
    wire [6:0]  l2_raddr = l2_rd_addr;
    wire [255:0] l2_rd_raw;

    l2_weights_bram l2_bram (
        .clka  (clk),           .ena   (1'b1),
        .wea   ({1{l2_we}}),    .addra (l2_waddr),  .dina (wr_data),
        .clkb  (clk),
        .addrb (l2_raddr),      .doutb (l2_rd_raw)
    );

    assign l2_weights = l2_rd_raw;

    always_ff @(posedge clk)
        l2_weights_r <= l2_weights;

    // -------------------------------------------------------
    // Layer 3 BRAM - 3 neurons
    // Write port: 8-bit data, 7-bit address
    // Read port:  32-bit data, 5-bit address
    // -------------------------------------------------------
    logic [6:0]             l3_rd_addr;
    logic [(H3_SIZE*8)-1:0] l3_weights;
    logic [(H3_SIZE*8)-1:0] l3_weights_r;

    wire l3_we = wr_en && (wr_bank == 2'd3);
    // neuron index: wr_addr[9:8] (2 bits, values 0..2)
    // input index:  wr_addr[4:0] (5 bits, values 0..32)
    wire [6:0] l3_waddr = {wr_addr[9:8], wr_addr[4:0]};
    wire [4:0] l3_raddr = l3_rd_addr[4:0];
    wire [31:0] l3_rd_raw;

    l3_weights_bram l3_bram (
        .clka  (clk),           .ena   (1'b1),
        .wea   ({1{l3_we}}),    .addra (l3_waddr),  .dina (wr_data),
        .clkb  (clk),
        .addrb (l3_raddr),      .doutb (l3_rd_raw)
    );

    // only 24 bits used (3 neurons x 8-bit weights); top byte is BRAM padding
    assign l3_weights = l3_rd_raw[23:0];

    always_ff @(posedge clk)
        l3_weights_r <= l3_weights;

    // -------------------------------------------------------
    // Layer 1 MAC - 64 neurons, Q1.15 inputs, 32-bit accumulators
    // -------------------------------------------------------
    logic               l1_start;
    logic signed [15:0] l1_in_val;
    logic signed [31:0] l1_out [H1_SIZE-1:0];
    logic               l1_done;

    mac_layer #(
        .N_INPUTS(N_FEATURES), .N_NEURONS(H1_SIZE),
        .APPLY_RELU(1), .IN_WIDTH(16), .ACC_WIDTH(32)
    ) l1_mac (
        .clk(clk), .resetn(resetn),
        .start(l1_start), .in_val(l1_in_val),
        .weights_flat(l1_weights_r),
        .n_inputs(8'(N_FEATURES)),
        .out(l1_out), .done(l1_done)
    );

    logic signed [31:0] l1_buffer [H1_SIZE-1:0];
    logic               l1_buf_valid;

    // -------------------------------------------------------
    // Layer 2 MAC - 32 neurons, 32-bit inputs, 48-bit accumulators
    // -------------------------------------------------------
    logic               l2_start;
    logic signed [31:0] l2_in_val;
    logic signed [47:0] l2_out [H2_SIZE-1:0];
    logic               l2_done;

    mac_layer #(
        .N_INPUTS(H1_SIZE), .N_NEURONS(H2_SIZE),
        .APPLY_RELU(1), .IN_WIDTH(32), .ACC_WIDTH(48)
    ) l2_mac (
        .clk(clk), .resetn(resetn),
        .start(l2_start), .in_val(l2_in_val),
        .weights_flat(l2_weights_r),
        .n_inputs(8'(H1_SIZE)),
        .out(l2_out), .done(l2_done)
    );

    logic signed [47:0] l2_buffer [H2_SIZE-1:0];

    // -------------------------------------------------------
    // Layer 3 MAC - 3 neurons, 32-bit inputs, 48-bit accumulators, no ReLU
    // -------------------------------------------------------
    logic               l3_start;
    logic signed [31:0] l3_in_val;
    logic signed [47:0] l3_out [H3_SIZE-1:0];
    logic               l3_done;

    mac_layer #(
        .N_INPUTS(H2_SIZE), .N_NEURONS(H3_SIZE),
        .APPLY_RELU(0), .IN_WIDTH(32), .ACC_WIDTH(48)
    ) l3_mac (
        .clk(clk), .resetn(resetn),
        .start(l3_start), .in_val(l3_in_val),
        .weights_flat(l3_weights_r),
        .n_inputs(8'(H2_SIZE)),
        .out(l3_out), .done(l3_done)
    );

    // -------------------------------------------------------
    // L1 early trigger - fires N_FEATURES+2 cycles before L2 finishes
    // so L1 can complete before L2 needs the next pixel's result
    // -------------------------------------------------------
    logic       l2_running;
    logic [6:0] l2_buf_idx;

    wire l1_early_trigger = l2_running && (l2_buf_idx == H1_SIZE - N_FEATURES - 2);
    assign l1_advance_lx_out = l1_early_trigger;

    // -------------------------------------------------------
    // Layer 1 control
    // rd_addr sent 2 cycles ahead: 1 for BRAM latency + 1 for l1_weights_r register
    // -------------------------------------------------------
    logic [4:0] l1_input_idx;
    logic       l1_running;
    logic       l1_stall;

    always_ff @(posedge clk) begin
        l1_start     <= 1'b0;
        l1_buf_valid <= 1'b0;
        l1_stall     <= 1'b0;

        if (!resetn) begin
            l1_running   <= 1'b0;
            l1_input_idx <= '0;
            l1_rd_addr   <= '0;
        end else begin
            if (start || l1_early_trigger)
                l1_stall <= 1'b1;

            if (l1_stall) begin
                l1_running   <= 1'b1;
                l1_input_idx <= '0;
                l1_rd_addr   <= '0;
            end

            if (l1_running) begin
                l1_rd_addr   <= l1_input_idx + 7'd2;
                l1_in_val    <= x[l1_input_idx];

                if (l1_input_idx == 0)
                    l1_start <= 1'b1;

                l1_input_idx <= l1_input_idx + 5'(1);

                if (l1_input_idx == N_FEATURES - 1)
                    l1_running <= 1'b0;
            end

            if (l1_done) begin
                for (int j = 0; j < H1_SIZE; j++)
                    l1_buffer[j] <= l1_out[j];
                l1_buf_valid <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------
    // Layer 2 control
    // rd_addr sent 2 cycles ahead: 1 for BRAM latency + 1 for l2_weights_r register
    // -------------------------------------------------------
    always_ff @(posedge clk) begin
        l2_start <= 1'b0;

        if (!resetn) begin
            l2_running <= 1'b0;
            l2_buf_idx <= '0;
            l2_rd_addr <= '0;
        end else begin
            if (l1_buf_valid) begin
                l2_running <= 1'b1;
                l2_buf_idx <= '0;
                l2_rd_addr <= '0;
            end

            if (l2_running) begin
                l2_rd_addr <= l2_buf_idx + 7'd2;
                l2_in_val  <= l1_buffer[l2_buf_idx];

                if (l2_buf_idx == 0)
                    l2_start <= 1'b1;

                l2_buf_idx <= l2_buf_idx + 7'(1);

                if (l2_buf_idx == H1_SIZE - 1)
                    l2_running <= 1'b0;
            end

            if (l2_done) begin
                for (int j = 0; j < H2_SIZE; j++)
                    l2_buffer[j] <= l2_out[j];
                l2_buf_idx <= '0;
            end
        end
    end

    // -------------------------------------------------------
    // Layer 3 control
    // rd_addr sent 2 cycles ahead: 1 for BRAM latency + 1 for l3_weights_r register
    // -------------------------------------------------------
    logic       l3_running;
    logic [5:0] l3_buf_idx;

    always_ff @(posedge clk) begin
        l3_start <= 1'b0;

        if (!resetn) begin
            l3_running <= 1'b0;
            l3_buf_idx <= '0;
            l3_rd_addr <= '0;
        end else begin
            if (l2_done) begin
                l3_running <= 1'b1;
                l3_buf_idx <= '0;
                l3_rd_addr <= '0;
            end

            if (l3_running) begin
                l3_rd_addr <= l3_buf_idx + 7'd2;
                // truncate 48-bit L2 accumulator to 32 bits - safe after ReLU clamps negatives to 0
                l3_in_val  <= l2_buffer[l3_buf_idx][31:0];

                if (l3_buf_idx == 0)
                    l3_start <= 1'b1;

                l3_buf_idx <= l3_buf_idx + 6'(1);

                if (l3_buf_idx == H2_SIZE - 1)
                    l3_running <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------
    // Normalisation - three pipeline stages to meet timing
    //
    // Stage 1 (l3_done):   subtract z_offset, register result
    // Stage 2 (l3_done_r): multiply by z_scale and shift right 16, register result
    // Stage 3 (l3_done_rr): clamp to [0,255] and output
    // valid pulses two cycles after l3_done
    // -------------------------------------------------------
    logic signed [47:0] r_sub, g_sub, b_sub;
    logic               l3_done_r;

    always_ff @(posedge clk) begin
        if (!resetn) begin
            l3_done_r <= 1'b0;
            r_sub     <= '0;
            g_sub     <= '0;
            b_sub     <= '0;
        end else begin
            l3_done_r <= l3_done;
            if (l3_done) begin
                r_sub <= l3_out[0] - z_offset[0];
                g_sub <= l3_out[1] - z_offset[1];
                b_sub <= l3_out[2] - z_offset[2];
            end
        end
    end

    // Stage 2 - register the multiply+shift result
    logic signed [63:0] r_norm, g_norm, b_norm;
    logic               l3_done_rr;

    always_ff @(posedge clk) begin
        if (!resetn) begin
            l3_done_rr <= 1'b0;
            r_norm     <= '0;
            g_norm     <= '0;
            b_norm     <= '0;
        end else begin
            l3_done_rr <= l3_done_r;
            if (l3_done_r) begin
                r_norm <= (r_sub * $signed({1'b0, z_scale[0]})) >>> 16;
                g_norm <= (g_sub * $signed({1'b0, z_scale[1]})) >>> 16;
                b_norm <= (b_sub * $signed({1'b0, z_scale[2]})) >>> 16;
            end
        end
    end

    // Stage 3 - clamp and output
    always_ff @(posedge clk) begin
        valid <= 1'b0;
        if (!resetn) begin
            r <= '0;
            g <= '0;
            b <= '0;
        end else if (l3_done_rr) begin
            r     <= (r_norm < 0) ? 8'd0 : (r_norm > 255) ? 8'd255 : r_norm[7:0];
            g     <= (g_norm < 0) ? 8'd0 : (g_norm > 255) ? 8'd255 : g_norm[7:0];
            b     <= (b_norm < 0) ? 8'd0 : (b_norm > 255) ? 8'd255 : b_norm[7:0];
            valid <= 1'b1;
        end
    end

endmodule