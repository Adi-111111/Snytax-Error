module mlp_pipeline #(
    parameter N_FEATURES = 16,
    parameter H1_SIZE    = 64,
    parameter H2_SIZE    = 32,
    parameter H3_SIZE    = 3
)(
    input                           clk,
    input                           resetn,

    // pixel input
    input                           start,
    input signed [15:0]             x [N_FEATURES-1:0],

    // weight RAM write port (from AXI)
    input [15:0]                    wr_addr,
    input [7:0]                     wr_data,
    input                           wr_en,
    input [1:0]                     wr_bank,

    // normalisation parameters
    input signed [31:0]             z_offset,
    input signed [31:0]             z_scale,

    // output
    output reg [7:0]                r, g, b,
    output reg                      valid,

    // fires one cycle before L1 restarts — pixel_generator uses this to advance lx/ly
    output wire                     l1_advance_lx_out
);

    integer j;

    // -------------------------------------------------------
    // Layer 1 RAM - 64 wide, 17 deep
    // -------------------------------------------------------
    reg [6:0]                       l1_rd_addr;
    wire [(H1_SIZE*8)-1:0]          l1_weights;

    weight_ram #(
        .N_NEURONS(H1_SIZE),
        .N_INPUTS(N_FEATURES)
    ) l1_ram (
        .clk(clk),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd0),
        .rd_addr(l1_rd_addr),
        .rd_data(l1_weights)
    );

    // -------------------------------------------------------
    // Layer 2 RAM - 128 wide, 65 deep, split odd/even
    // -------------------------------------------------------
    reg [6:0]                       l2_rd_addr;
    wire [(64*8)-1:0]               l2_weights_even;
    wire [(64*8)-1:0]               l2_weights_odd;
    wire [(128*8)-1:0]              l2_weights = {l2_weights_odd, l2_weights_even};

    weight_ram #(
        .N_NEURONS(64),
        .N_INPUTS(H1_SIZE)
    ) l2_ram_even (
        .clk(clk),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd1),
        .rd_addr(l2_rd_addr),
        .rd_data(l2_weights_even)
    );

    weight_ram #(
        .N_NEURONS(64),
        .N_INPUTS(H1_SIZE)
    ) l2_ram_odd (
        .clk(clk),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd2),
        .rd_addr(l2_rd_addr),
        .rd_data(l2_weights_odd)
    );

    // -------------------------------------------------------
    // Layer 3 RAM - 8 wide, 33 deep
    // -------------------------------------------------------
    reg [6:0]                       l3_rd_addr;
    wire [(8*8)-1:0]                l3_weights;

    weight_ram #(
        .N_NEURONS(8),
        .N_INPUTS(H2_SIZE)
    ) l3_ram (
        .clk(clk),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .wr_en(wr_en && wr_bank == 2'd3),
        .rd_addr(l3_rd_addr),
        .rd_data(l3_weights)
    );

    // -------------------------------------------------------
    // Layer 1 MAC - 64 neurons, 16-bit input, 32-bit accumulator
    // -------------------------------------------------------
    reg                             l1_start;
    reg signed [15:0]               l1_in_val;
    wire signed [31:0]              l1_out [H1_SIZE-1:0];
    wire                            l1_done;

    mac_layer #(
        .N_INPUTS(N_FEATURES),
        .N_NEURONS(H1_SIZE),
        .APPLY_RELU(1),
        .IN_WIDTH(16),
        .ACC_WIDTH(32)
    ) l1_mac (
        .clk(clk),
        .resetn(resetn),
        .start(l1_start),
        .in_val(l1_in_val),
        .weights_flat(l1_weights),
        .n_inputs(8'(N_FEATURES)),
        .out(l1_out),
        .done(l1_done)
    );

    // -------------------------------------------------------
    // Layer 1 -> Layer 2 buffer
    // -------------------------------------------------------
    reg signed [31:0]               l1_buffer [H1_SIZE-1:0];
    reg [6:0]                       l1_buf_idx;
    reg                             l1_buf_valid;

    // -------------------------------------------------------
    // Layer 2 MAC - 128 DSPs, 32-bit input, 48-bit accumulator
    // -------------------------------------------------------
    reg                             l2_start;
    reg signed [31:0]               l2_in_val;
    wire signed [47:0]              l2_out [128-1:0];
    wire                            l2_done;

    mac_layer #(
        .N_INPUTS(H1_SIZE),
        .N_NEURONS(128),
        .APPLY_RELU(1),
        .IN_WIDTH(32),
        .ACC_WIDTH(48)
    ) l2_mac (
        .clk(clk),
        .resetn(resetn),
        .start(l2_start),
        .in_val(l2_in_val),
        .weights_flat(l2_weights),
        .n_inputs(8'(H1_SIZE)),
        .out(l2_out),
        .done(l2_done)
    );

    // -------------------------------------------------------
    // Layer 2 -> Layer 3 buffer
    // -------------------------------------------------------
    reg signed [47:0]               l2_buffer [H2_SIZE-1:0];
    reg [5:0]                       l2_buf_idx;

    // -------------------------------------------------------
    // Layer 3 MAC - 8 DSPs, 32-bit input, 48-bit accumulator
    // -------------------------------------------------------
    reg                             l3_start;
    reg signed [31:0]               l3_in_val;
    wire signed [47:0]              l3_out [8-1:0];
    wire                            l3_done;

    mac_layer #(
        .N_INPUTS(H2_SIZE),
        .N_NEURONS(8),
        .APPLY_RELU(0),
        .IN_WIDTH(32),
        .ACC_WIDTH(48)
    ) l3_mac (
        .clk(clk),
        .resetn(resetn),
        .start(l3_start),
        .in_val(l3_in_val),
        .weights_flat(l3_weights),
        .n_inputs(8'(H2_SIZE)),
        .out(l3_out),
        .done(l3_done)
    );

    // -------------------------------------------------------
    // Layer 1 control
    // -------------------------------------------------------
    reg [4:0]   l1_input_idx;
    reg         l1_running;
    reg         l1_stall;

    // declared here (before L1 always block) so l1_early_trigger can reference it;
    // driven by the L2 always block below
    reg         l2_running;

    // fires N_FEATURES+2 cycles before L2 finishes — L1 with 1-cycle stall
    // completes exactly when L2 feeds its last input (~64-cycle throughput)
    wire l1_early_trigger = l2_running && (l2_buf_idx == H1_SIZE - N_FEATURES - 2);
    assign l1_advance_lx_out = l1_early_trigger;

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
            // 1-cycle stall between trigger and l1_running so input_builder
            // has time to register the updated lx/ly from pixel_generator
            if (start || l1_early_trigger)
                l1_stall <= 1;

            if (l1_stall) begin
                l1_running   <= 1;
                l1_input_idx <= 0;
                l1_rd_addr   <= 0;
            end

            if (l1_running) begin
                l1_rd_addr   <= l1_input_idx + 1;
                l1_in_val    <= x[l1_input_idx];

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
                l1_buf_idx   <= 0;
            end
        end
    end

    // -------------------------------------------------------
    // Layer 2 control  (l2_running declared above near L1 control)
    // -------------------------------------------------------

    always @(posedge clk) begin
        l2_start <= 0;

        if (!resetn) begin
            l2_running <= 0;
            l2_buf_idx <= 0;
            l2_rd_addr <= 0;
        end

        else begin
            if (l1_buf_valid) begin
                l2_running <= 1;
                l2_buf_idx <= 0;
                l2_rd_addr <= 0;
            end

            if (l2_running) begin
                l2_rd_addr <= l2_buf_idx + 1;
                l2_in_val  <= l1_buffer[l2_buf_idx];

                if (l2_buf_idx == 0)
                    l2_start <= 1;

                l2_buf_idx <= l2_buf_idx + 1;

                if (l2_buf_idx == H1_SIZE - 1)
                    l2_running <= 0;
            end

            if (l2_done) begin
                for (j = 0; j < H2_SIZE; j = j + 1)
                    l2_buffer[j] <= l2_out[j];
                l2_buf_idx <= 0;
            end
        end
    end

    // -------------------------------------------------------
    // Layer 3 control
    // -------------------------------------------------------
    reg l3_running;
    reg [5:0] l3_buf_idx;

    always @(posedge clk) begin
        l3_start <= 0;
        valid    <= 0;

        if (!resetn) begin
            l3_running <= 0;
            l3_buf_idx <= 0;
            l3_rd_addr <= 0;
        end

        else begin
            if (l2_done) begin
                l3_running <= 1;
                l3_buf_idx <= 0;
                l3_rd_addr <= 0;
            end

            if (l3_running) begin
                l3_rd_addr <= l3_buf_idx + 1;
                l3_in_val  <= l2_buffer[l3_buf_idx];

                if (l3_buf_idx == 0)
                    l3_start <= 1;

                l3_buf_idx <= l3_buf_idx + 1;

                if (l3_buf_idx == H2_SIZE - 1)
                    l3_running <= 0;
            end

            if (l3_done) begin
                valid <= 1;
                begin : norm
                    reg signed [63:0] r_norm, g_norm, b_norm;
                    r_norm = ((l3_out[0] - z_offset) * z_scale) >> 16;
                    g_norm = ((l3_out[1] - z_offset) * z_scale) >> 16;
                    b_norm = ((l3_out[2] - z_offset) * z_scale) >> 16;

                    r <= (r_norm < 0) ? 8'd0 : (r_norm > 255) ? 8'd255 : r_norm[7:0];
                    g <= (g_norm < 0) ? 8'd0 : (g_norm > 255) ? 8'd255 : g_norm[7:0];
                    b <= (b_norm < 0) ? 8'd0 : (b_norm > 255) ? 8'd255 : b_norm[7:0];
                end
            end
        end
    end

    always @(posedge clk) begin
        if (l1_done) $display("MLP: l1_done fired");
        if (l2_done) $display("MLP: l2_done fired");
        if (l3_done) $display("MLP: l3_done fired");
        if (l1_buf_valid) $display("MLP: l1_buf_valid fired");
    end

endmodule