`timescale 1ns / 1ps
// tb_pixel_generator.v
// Verifies correct RGB output by tapping debug ports on pixel_generator
// that expose mlp_r, mlp_g, mlp_b directly (bypassing packer byte-interleaving).
// Uses AXI_LITE_ADDR_WIDTH=9 to reach weight registers at indices 69-71.
// Holds periph_resetn low during weight loading so pipeline starts clean.

module tb_pixel_generator;

    localparam N_FEATURES = 16;
    localparam H1_SIZE    = 64;
    localparam H2_SIZE    = 32;
    localparam H3_SIZE    = 3;

    localparam signed [7:0] W_HALF = 8'sd8;  // 0.5 in Q4.4

    // axis_x_select=14, axis_y_select=15 so pixel(0,0) maps features 14,15
    // to x_norm=0 and y_norm=0. Features 0-13 stay at default 0.5.
    // L1 per neuron = 14 × (0.5 × 0.5) = 3.5 → large enough to saturate to 255
    localparam EXP_R = 8'd255;
    localparam EXP_G = 8'd255;
    localparam EXP_B = 8'd255;

    // -----------------------------------------------------------------------
    // Signals
    // -----------------------------------------------------------------------
    reg         clk        = 0;
    reg         axi_rst    = 0;
    reg         periph_rst = 0;

    reg  [8:0]  awaddr  = 0;
    reg         awvalid = 0;
    wire        awready;
    reg  [31:0] wdata   = 0;
    reg         wvalid  = 0;
    wire        wready;
    reg         bready  = 1;
    wire [1:0]  bresp;
    wire        bvalid;

    wire [31:0] tdata;
    wire [3:0]  tkeep;
    wire        tlast;
    wire        tvalid;
    wire [0:0]  tuser;
    reg         tready  = 1;

    wire        arready;
    wire [31:0] rdata;
    wire        rvalid;
    wire [1:0]  rresp;

    // Debug ports — direct access to MLP output before packer
    wire [7:0]  dbg_r, dbg_g, dbg_b;
    wire        dbg_valid;

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    always #5 clk = !clk;

    // -----------------------------------------------------------------------
    // DUT — AXI_LITE_ADDR_WIDTH=9, debug ports enabled
    // -----------------------------------------------------------------------
    pixel_generator #(
        .AXI_LITE_ADDR_WIDTH(9),
        .DEBUG_PORTS(1)
    ) dut (
        .out_stream_aclk(clk),
        .s_axi_lite_aclk(clk),
        .axi_resetn(axi_rst),
        .periph_resetn(periph_rst),

        .out_stream_tdata(tdata),
        .out_stream_tkeep(tkeep),
        .out_stream_tlast(tlast),
        .out_stream_tready(tready),
        .out_stream_tvalid(tvalid),
        .out_stream_tuser(tuser),

        .s_axi_lite_araddr(9'h0),
        .s_axi_lite_arready(arready),
        .s_axi_lite_arvalid(1'b0),

        .s_axi_lite_awaddr(awaddr),
        .s_axi_lite_awready(awready),
        .s_axi_lite_awvalid(awvalid),

        .s_axi_lite_bready(bready),
        .s_axi_lite_bresp(bresp),
        .s_axi_lite_bvalid(bvalid),

        .s_axi_lite_rdata(rdata),
        .s_axi_lite_rready(1'b0),
        .s_axi_lite_rresp(rresp),
        .s_axi_lite_rvalid(rvalid),

        .s_axi_lite_wdata(wdata),
        .s_axi_lite_wready(wready),
        .s_axi_lite_wvalid(wvalid),

        .dbg_r(dbg_r),
        .dbg_g(dbg_g),
        .dbg_b(dbg_b),
        .dbg_valid(dbg_valid)
    );

    // -----------------------------------------------------------------------
    // AXI-Lite write task — blocking assignments for correct timing
    // -----------------------------------------------------------------------
    task axi_write;
        input [8:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk); #1;
            awaddr  = addr;
            awvalid = 1;
            wdata   = data;
            wvalid  = 1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            #1;
            awvalid = 0;
            wvalid  = 0;
            while (!bvalid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    // -----------------------------------------------------------------------
    // Weight write task
    // regfile[69]=weight_addr @ 0x114, [70]=weight_data @ 0x118, [71]=weight_we @ 0x11C
    // weight_addr[15:14]: 2'b00=L1, 2'b01=L2, 2'b10=L3
    // -----------------------------------------------------------------------
    task write_weight;
        input [15:0] waddr;
        input [7:0]  wdat;
        begin
            axi_write(9'h114, {16'h0, waddr});
            axi_write(9'h118, {24'h0, wdat});
            axi_write(9'h11C, 32'h1);
            axi_write(9'h11C, 32'h0);
        end
    endtask

    // -----------------------------------------------------------------------
    // Capture first pixel via debug ports (bypasses packer byte ordering)
    // -----------------------------------------------------------------------
    reg [7:0]  cap_r = 0, cap_g = 0, cap_b = 0;
    reg        pixel_captured = 0;

    always @(posedge clk) begin
        if (dbg_valid && !pixel_captured) begin
            cap_r <= dbg_r;
            cap_g <= dbg_g;
            cap_b <= dbg_b;
            pixel_captured <= 1;
        end
    end

    // -----------------------------------------------------------------------
    // Timeout guard — fail if first pixel never arrives
    // -----------------------------------------------------------------------
    initial begin
        #1000000000;  // 10ms timeout
        if (!pixel_captured) begin
            $display("TIMEOUT: first pixel never produced");
            $finish;
        end
    end

    // -----------------------------------------------------------------------
    // Stimulus
    // -----------------------------------------------------------------------
    integer n, j;

    initial begin
        $dumpfile("tb_pg.vcd");
        $dumpvars(0, tb_pixel_generator);

        // Release AXI reset, hold pixel pipeline
        #20; axi_rst = 1;
        #20;

        // Set axis selects so features 14,15 map to pixel coords at (0,0)=0
        // regfile[64]=axis_x_select @ 0x100
        // regfile[65]=axis_y_select @ 0x104
        $display("Setting axis selects to 14, 15...");
        axi_write(9'h100, 32'd14);
        axi_write(9'h104, 32'd15);

        $display("=== Loading weights ===");

        // L1: 64 neurons × 16 inputs, all = 0.5 (Q4.4=8), wr_bank=0
        $display("L1 (%0d writes)...", H1_SIZE * N_FEATURES);
        for (n = 0; n < H1_SIZE; n = n + 1)
            for (j = 0; j < N_FEATURES; j = j + 1)
                write_weight((n << 8) | j, W_HALF);

        // L2: 32 neurons × 64 inputs, all = 0.5, wr_bank=1 → base 0x4000
        $display("L2 (%0d writes)...", H2_SIZE * H1_SIZE);
        for (n = 0; n < H2_SIZE; n = n + 1)
            for (j = 0; j < H1_SIZE; j = j + 1)
                write_weight(16'h4000 | (n << 8) | j, W_HALF);

        // L3: 3 neurons × 32 inputs, all = 0.5, wr_bank=2 → base 0x8000
        $display("L3 (%0d writes)...", H3_SIZE * H2_SIZE);
        for (n = 0; n < H3_SIZE; n = n + 1)
            for (j = 0; j < H2_SIZE; j = j + 1)
                write_weight(16'h8000 | (n << 8) | j, W_HALF);

        $display("=== Releasing pixel pipeline ===");
        @(posedge clk); #1;
        periph_rst = 1;

        #100;
        $display("DEBUG pipeline: l1_running=%b l1_start=%b l1_buf_valid=%b",
            dut.mlp.l1_running, dut.mlp.l1_start, dut.mlp.l1_buf_valid);
        $display("DEBUG pipeline: l2_running=%b l3_running=%b",
            dut.mlp.l2_running, dut.mlp.l3_running);
        $display("DEBUG RAM L1: mem[0][0]=%h mem[1][0]=%h",
            dut.mlp.l1_ram.mem[0][0], dut.mlp.l1_ram.mem[1][0]);
        $display("DEBUG RAM L2b0: mem[0][0]=%h",
    dut.mlp.gen_l2_ram[0].l2_ram_inst.mem[0][0]);
        $display("DEBUG: first_start=%b mlp_start=%b periph_rst=%b",
            dut.first_start, dut.mlp_start, periph_rst);

        $display("Waiting for first pixel...");
        wait(pixel_captured);
        #10;

        $display("=== Pixel (0,0) result ===");
        $display("  R = %3d  expected %3d  %s", cap_r, EXP_R,
                  cap_r == EXP_R ? "PASS" : "FAIL");
        $display("  G = %3d  expected %3d  %s", cap_g, EXP_G,
                  cap_g == EXP_G ? "PASS" : "FAIL");
        $display("  B = %3d  expected %3d  %s", cap_b, EXP_B,
                  cap_b == EXP_B ? "PASS" : "FAIL");

        if (cap_r == EXP_R && cap_g == EXP_G && cap_b == EXP_B)
            $display("OVERALL: PASS");
        else
            $display("OVERALL: FAIL");

        // check a specific weight was written
        $display("DEBUG RAM: l1_ram mem[0][0]=%h", dut.mlp.l1_ram.mem[0][0]);
        $display("DEBUG RAM: l2_ram_b0 mem[0][0]=%h", dut.mlp.gen_l2_ram[0].l2_ram_inst.mem[0][0]);
        
        $display("DEBUG: dbg_valid=%b mlp_r=%h mlp_g=%h mlp_b=%h", dbg_valid, dbg_r, dbg_g, dbg_b);
        $display("DEBUG: regfile[64]=%0d regfile[65]=%0d", 
            dut.regfile[64], dut.regfile[65]);

        #100 $finish;
    end

endmodule