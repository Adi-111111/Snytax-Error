
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
    input  [15:0]                   wr_addr,
    input  [7:0]                    wr_data,
    input                           wr_en,
    input  [1:0]                    wr_bank,
    input  signed [31:0]            z_offset,
    input  [31:0]                   z_scale,
    input  [4:0]                    z_shift,
    output reg [7:0]                r, g, b,
    output reg                      valid,
    output wire                     l1_advance_lx_out
);

    integer j;
    reg signed [63:0] r_norm, g_norm, b_norm;
    reg signed [47:0] l3_shifted  [H3_SIZE-1:0];
    reg signed [47:0] l3_diff     [H3_SIZE-1:0]; 
    reg signed [63:0] l3_scaled   [H3_SIZE-1:0]; 
    reg               l3_shift_valid;
    reg               l3_diff_valid;
    reg               l3_scaled_valid;
    localparam signed [15:0] BIAS_CONST_L1  = 16'sd32767; 
    localparam signed [31:0] BIAS_CONST_L23 = 32'sd32767; 
    wire [15:0] wr_addr_masked = {2'b00, wr_addr[13:0]};

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

    reg  [6:0]                      l2_rd_addr;
    wire [(H2_SIZE*8)-1:0]          l2_weights_flat;

    genvar bk;

    weight_ram #(
        .N_NEURONS(H2_SIZE), .N_INPUTS(H1_SIZE),
        .N_BANKS(1), .BANK_ID(0)
    ) l2_ram (
        .clk(clk),
        .wr_addr(wr_addr_masked), .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd1),
        .rd_addr(l2_rd_addr),
        .rd_data(l2_weights_flat)
    );
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
        .n_inputs(7'(N_FEATURES + 1)),  
        .out(l1_out), .done(l1_done)
    );

    reg signed [31:0]               l1_buffer [H1_SIZE-1:0];
    reg                             l1_buf_valid;

    reg                             l2_start;
    reg  signed [17:0]              l2_in_val_flat;
    wire signed [47:0]              l2_out [H2_SIZE-1:0]; 
    wire                            l2_done;

    function automatic signed [17:0] clamp18(input signed [31:0] v);
        begin
            if (v > 32'sd131071)
                clamp18 = 18'sd131071;
            else if (v < -32'sd131072)
                clamp18 = -18'sd131072;
            else
                clamp18 = v[17:0];
        end
    endfunction

    mac_layer #(
        .N_INPUTS(H1_SIZE),
        .N_NEURONS(H2_SIZE),
        .APPLY_RELU(1),
        .IN_WIDTH(18), .ACC_WIDTH(48), .N_PARALLEL(1), .USE_DSP(0)
    ) l2_mac (
        .clk(clk), .resetn(resetn),
        .start(l2_start),
        .in_val_flat(l2_in_val_flat),
        .weights_flat(l2_weights_flat),
        .n_inputs(7'(H1_SIZE + 1)),  
        .out(l2_out), .done(l2_done)
    );

    reg signed [47:0]               l2_buffer [H2_SIZE-1:0];

    reg                             l3_start;
    reg  signed [(2*32)-1:0]        l3_in_val_flat;
    wire signed [47:0]              l3_out [H3_SIZE-1:0];
    wire                            l3_done;

    mac_layer #(
        .N_INPUTS(H2_SIZE),   // full count — MAC divides by N_PARALLEL internally
        .N_NEURONS(H3_SIZE),  // 3 actual output neurons
        .APPLY_RELU(0),
        .IN_WIDTH(32), .ACC_WIDTH(48), .N_PARALLEL(2)
    ) l3_mac (
        .clk(clk), .resetn(resetn),
        .start(l3_start),
        .in_val_flat(l3_in_val_flat),
        .weights_flat(l3_weights_flat),
        .n_inputs(7'(H2_SIZE + 1)),   // +1 for bias cycle (17 cycles total)
        .out(l3_out), .done(l3_done)
    );

    wire l1_early_trigger = l2_done;
    assign l1_advance_lx_out = l1_early_trigger;

    reg [4:0]   l1_input_idx;
    reg         l1_running;
    reg         l1_stall;
    reg         l1_stall2; 

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

            if (l1_stall) begin
                l1_stall2    <= 1;
                l1_input_idx <= 0;
                l1_rd_addr   <= 0;
            end

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

    reg         l2_running;
    reg         l2_init;    // extra pre-load cycle for 2-stage BRAM read
    reg [6:0]   l2_in_idx;
    reg [6:0]   l2_rd_cycle;  // needs to count to H1_SIZE = 64

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

                // bias cycle: inject constant 1.0 instead of an L1 activation.
                // Bias weights live at global column H1_SIZE.
                if (l2_rd_cycle == H1_SIZE)
                    l2_in_val_flat <= BIAS_CONST_L23[17:0];
                else begin
                    l2_in_val_flat <= clamp18(l1_buffer[l2_in_idx]);
                    l2_in_idx <= l2_in_idx + 1;
                end

                if (l2_rd_cycle == 0)
                    l2_start <= 1;

                l2_rd_cycle <= l2_rd_cycle + 1;

                if (l2_rd_cycle == H1_SIZE)  // one extra cycle for bias
                    l2_running <= 0;
            end

            if (l2_done) begin
                // capture all H2_SIZE=32 actual neuron outputs
                for (j = 0; j < H2_SIZE; j = j + 1)
                    l2_buffer[j] <= l2_out[j];
            end
        end
    end

    reg         l3_running;
    reg         l3_init;    // extra pre-load cycle for 2-stage BRAM read
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

                // bias cycle: lane 0 = BIAS_CONST, lane 1 = 0
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

            // Stage 1: arithmetic right-shift by z_shift — collapses ±500B accumulator
            if (l3_done) begin
                for (j = 0; j < H3_SIZE; j = j + 1)
                    l3_shifted[j] <= $signed(l3_out[j]) >>> z_shift;
                l3_shift_valid <= 1;
            end

            // Stage 2: subtract z_offset
            if (l3_shift_valid) begin
                for (j = 0; j < H3_SIZE; j = j + 1)
                    l3_diff[j] <= l3_shifted[j] - z_offset;
                l3_diff_valid <= 1;
            end

            // Stage 3: multiply by z_scale — $signed cast keeps negative l3_diff correct
            if (l3_diff_valid) begin
                for (j = 0; j < H3_SIZE; j = j + 1)
                    l3_scaled[j] <= $signed(l3_diff[j]) * $signed({1'b0, z_scale});
                l3_scaled_valid <= 1;
            end

            // Stage 4: >>>16 + clamp → RGB (arithmetic shift preserves sign for negative channels)
            if (l3_scaled_valid) begin
                valid  <= 1;
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