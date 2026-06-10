`timescale 1ns / 1ps
module pixgen_tb;

parameter REG_COUNT = 8;

reg clk = 0;
reg rst = 0;
always #5 clk = !clk;

// AXI Stream -- connected but not the focus of this test
wire [31:0] dataOut;
wire [3:0]  keep;
wire        last, valid, user;

// AXI-Lite write address channel
reg  [8:0]  writeAdd      = 0;   // 9-bit to match AXI_LITE_ADDR_WIDTH=9
reg  [31:0] writeData     = 0;
reg         writeAddValid = 0;
reg         writeValid    = 0;
wire        writeAddReady;
wire        writeReady;

// AXI-Lite read address channel
reg  [8:0]  readAdd       = 0;   // 9-bit
wire        readAddReady;
reg         readAddValid  = 0;

// AXI-Lite read data channel
wire [31:0] readData;
reg         readReady     = 0;
wire [1:0]  readResp;
wire        readValid;

// AXI-Lite write response channel
reg         respReady     = 0;
wire [1:0]  respData;
wire        respValid;

// debug outputs -- open (DUT drives 0 when DEBUG_PORTS=0)
wire [7:0] dbg_r, dbg_g, dbg_b;
wire       dbg_valid;
wire [9:0] dbg_x;
wire [8:0] dbg_y;

pixel_generator p1 (
  .out_stream_aclk(clk),
  .s_axi_lite_aclk(clk),
  .axi_resetn(rst),
  .periph_resetn(rst),

  .out_stream_tdata(dataOut),
  .out_stream_tkeep(keep),
  .out_stream_tlast(last),
  .out_stream_tready(1'b1),
  .out_stream_tvalid(valid),
  .out_stream_tuser(user),

  .s_axi_lite_araddr(readAdd),
  .s_axi_lite_arready(readAddReady),
  .s_axi_lite_arvalid(readAddValid),

  .s_axi_lite_awaddr(writeAdd),
  .s_axi_lite_awready(writeAddReady),
  .s_axi_lite_awvalid(writeAddValid),

  .s_axi_lite_bready(respReady),
  .s_axi_lite_bresp(respData),
  .s_axi_lite_bvalid(respValid),

  .s_axi_lite_rdata(readData),
  .s_axi_lite_rready(readReady),
  .s_axi_lite_rresp(readResp),
  .s_axi_lite_rvalid(readValid),

  .s_axi_lite_wdata(writeData),
  .s_axi_lite_wready(writeReady),
  .s_axi_lite_wvalid(writeValid),

  .dbg_r(dbg_r), .dbg_g(dbg_g), .dbg_b(dbg_b),
  .dbg_valid(dbg_valid),
  .dbg_x(dbg_x), .dbg_y(dbg_y)
);

// ---------------------------------------------------------------------------
// AXI-Lite write task
// Both AW and W are driven simultaneously. The DUT accepts both in one cycle
// when idle (awready && wready both high in AWAIT_WADD_AND_DATA state).
// We hold both valids until both readys are seen on the same posedge, then
// de-assert and complete the B-channel handshake.
// ---------------------------------------------------------------------------
task axi_write;
    input [8:0]  addr;
    input [31:0] data;
    begin
        @(negedge clk);
        writeAdd      = addr;
        writeData     = data;
        writeAddValid = 1;
        writeValid    = 1;

        // hold both valids until both readys are simultaneously high
        @(posedge clk);
        while (!(writeAddReady && writeReady)) @(posedge clk);
        @(negedge clk);
        writeAddValid = 0;
        writeValid    = 0;

        // wait for B-channel: assert bready once bvalid appears
        @(posedge clk);
        while (!respValid) @(posedge clk);
        @(negedge clk); respReady = 1;
        @(posedge clk);   // bvalid & bready both high -- handshake done
        @(negedge clk); respReady = 0;
    end
endtask

// ---------------------------------------------------------------------------
// AXI-Lite read task
// Holds arvalid until arready, then waits for rvalid and samples rdata.
// ---------------------------------------------------------------------------
task axi_read;
    input  [8:0]  addr;
    output [31:0] data;
    begin
        @(negedge clk);
        readAdd      = addr;
        readAddValid = 1;

        @(posedge clk);
        while (!readAddReady) @(posedge clk);
        @(negedge clk); readAddValid = 0;

        // wait for rvalid; assert rready at negedge so it's stable before the
        // next posedge where the DUT samples it and transitions to AWAIT_RADD
        @(posedge clk);
        while (!readValid) @(posedge clk);
        @(negedge clk); readReady = 1;
        @(posedge clk);                // rvalid=1, rready=1: handshake done
        data = readData;               // readData stable from previous cycle's NBA
        @(negedge clk); readReady = 0;
    end
endtask

integer      i;
integer      errors = 0;
reg [31:0]   rdback;
reg [31:0]   expected;

initial begin
    repeat(3) @(posedge clk);
    rst = 1;
    repeat(2) @(posedge clk);

    // -- Write REG_COUNT general registers ----------------------------------
    $display("=== AXI-Lite write (regs 0..%0d) ===", REG_COUNT-1);
    for (i = 0; i < REG_COUNT; i = i+1) begin
        axi_write(i*4, i * 32'h11111111);
        $display("  wrote reg[%0d] = %08h", i, i * 32'h11111111);
    end

    // -- Read back and verify -----------------------------------------------
    $display("=== AXI-Lite read-back ===");
    for (i = 0; i < REG_COUNT; i = i+1) begin
        axi_read(i*4, rdback);
        expected = i * 32'h11111111;
        if (rdback !== expected) begin
            $display("  FAIL reg[%0d]: expected=%08h  got=%08h", i, expected, rdback);
            errors = errors + 1;
        end else
            $display("  PASS reg[%0d] = %08h", i, rdback);
    end

    // -- Weight register path (byte addr = reg_idx*4; reg 64 = addr 0x100) --
    // Weight regs require 9-bit address (bit 8 set); unreachable with 8-bit addr.
    $display("=== Weight register write/read (reg 64 @ addr 0x100) ===");
    axi_write(9'h100, 32'hDEAD_BEEF);
    axi_read (9'h100, rdback);
    if (rdback !== 32'hDEAD_BEEF) begin
        $display("  FAIL reg[64]: expected=deadbeef  got=%08h", rdback);
        errors = errors + 1;
    end else
        $display("  PASS reg[64] = %08h", rdback);

    axi_write(9'h114, 32'h0000_3C0F);   // reg 69 -- weight_addr
    axi_write(9'h118, 32'h0000_007F);   // reg 70 -- weight_data (int8 = 127)
    axi_write(9'h11C, 32'h0000_0001);   // reg 71 -- assert weight_we
    axi_write(9'h11C, 32'h0000_0000);   // reg 71 -- clear weight_we

    axi_read(9'h114, rdback);
    if (rdback !== 32'h0000_3C0F) begin
        $display("  FAIL reg[69] weight_addr: expected=00003c0f  got=%08h", rdback);
        errors = errors + 1;
    end else
        $display("  PASS reg[69] weight_addr = %08h", rdback);

    // -- Summary ------------------------------------------------------------
    if (errors == 0)
        $display("All AXI-Lite handshake tests PASSED.");
    else
        $display("%0d test(s) FAILED.", errors);

    #50 $finish;
end

endmodule
