module pixel_generator #(
    parameter AXI_LITE_ADDR_WIDTH = 9,
    parameter REG_FILE_SIZE = 75,
    parameter DEBUG_PORTS = 0
)(
    input           out_stream_aclk,
    input           s_axi_lite_aclk,
    input           axi_resetn,
    input           periph_resetn,

    output [31:0]   out_stream_tdata,
    output [3:0]    out_stream_tkeep,
    output          out_stream_tlast,
    input           out_stream_tready,
    output          out_stream_tvalid,
    output [0:0]    out_stream_tuser,

    input [AXI_LITE_ADDR_WIDTH-1:0]     s_axi_lite_araddr,
    output          s_axi_lite_arready,
    input           s_axi_lite_arvalid,

    input [AXI_LITE_ADDR_WIDTH-1:0]     s_axi_lite_awaddr,
    output          s_axi_lite_awready,
    input           s_axi_lite_awvalid,

    input           s_axi_lite_bready,
    output [1:0]    s_axi_lite_bresp,
    output          s_axi_lite_bvalid,

    output [31:0]   s_axi_lite_rdata,
    input           s_axi_lite_rready,
    output [1:0]    s_axi_lite_rresp,
    output          s_axi_lite_rvalid,

    input  [31:0]   s_axi_lite_wdata,
    output          s_axi_lite_wready,
    input           s_axi_lite_wvalid,

    // debug outputs - only used in testbench (DEBUG_PORTS=1)
    output [7:0]    dbg_r,
    output [7:0]    dbg_g,
    output [7:0]    dbg_b,
    output          dbg_valid,
    output [9:0]    dbg_x,
    output [8:0]    dbg_y
);

    localparam X_SIZE = 640;
    localparam Y_SIZE = 480;
    localparam N_FEATURES = 16;
    localparam H1_SIZE    = 64;
    localparam H2_SIZE    = 32;
    localparam REG_FILE_AWIDTH = 7;

    localparam AWAIT_WADD_AND_DATA = 3'b000;
    localparam AWAIT_WDATA         = 3'b001;
    localparam AWAIT_WADD          = 3'b010;
    localparam AWAIT_WRITE         = 3'b100;
    localparam AWAIT_RESP          = 3'b101;

    localparam AWAIT_RADD  = 2'b00;
    localparam AWAIT_FETCH = 2'b01;
    localparam AWAIT_READ  = 2'b10;

    localparam AXI_OK  = 2'b00;
    localparam AXI_ERR = 2'b10;

    // Register file
    reg [31:0] regfile [REG_FILE_SIZE-1:0];
    reg [REG_FILE_AWIDTH-1:0] writeAddr, readAddr;
    reg [31:0]                readData, writeData;
    reg [1:0]                 readState  = AWAIT_RADD;
    reg [2:0]                 writeState = AWAIT_WADD_AND_DATA;

    integer ri;
    initial begin
        for (ri = 0; ri < REG_FILE_SIZE; ri = ri + 1)
            regfile[ri] = 0;
        for (ri = 0; ri < 16; ri = ri + 1)
            regfile[ri] = 32'd16384;
        regfile[64] = 32'd0;
        regfile[65] = 32'd1;
        regfile[66] = 32'd16;
        regfile[67] = 32'd64;
        regfile[68] = 32'd32;
        regfile[71] = 32'd0;
        regfile[72] = 32'd0;
        regfile[73] = 32'd65536;
        regfile[74] = 32'd16;
    end

    wire signed [15:0] feature_regs [N_FEATURES-1:0];
    genvar fi;
    generate
        for (fi = 0; fi < N_FEATURES; fi = fi + 1) begin
            assign feature_regs[fi] = regfile[fi][15:0];
        end
    endgenerate

    wire [5:0]  axis_x_select = regfile[64][5:0];
    wire [5:0]  axis_y_select = regfile[65][5:0];
    wire [15:0] weight_addr   = regfile[69][15:0];
    wire [7:0]  weight_data   = regfile[70][7:0];
    wire        weight_we     = regfile[71][0];
    wire signed [31:0] z_offset = regfile[72];
    wire [31:0] z_scale        = regfile[73];
    wire [4:0]  z_shift        = regfile[74][4:0];

    // AXI-Lite read state machine
    always @(posedge s_axi_lite_aclk) begin
        readData <= regfile[readAddr];
        if (!axi_resetn) begin
            readState <= AWAIT_RADD;
        end
        else case (readState)
            AWAIT_RADD: begin
                if (s_axi_lite_arvalid) begin
                    readAddr  <= s_axi_lite_araddr[2+:REG_FILE_AWIDTH];
                    readState <= AWAIT_FETCH;
                end
            end
            AWAIT_FETCH: readState <= AWAIT_READ;
            AWAIT_READ: begin
                if (s_axi_lite_rready) readState <= AWAIT_RADD;
            end
            default: readState <= AWAIT_RADD;
        endcase
    end

    assign s_axi_lite_arready = (readState == AWAIT_RADD);
    assign s_axi_lite_rresp   = (readAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;
    assign s_axi_lite_rvalid  = (readState == AWAIT_READ);
    assign s_axi_lite_rdata   = readData;

    // AXI-Lite write state machine
    always @(posedge s_axi_lite_aclk) begin
        if (!axi_resetn) begin
            writeState <= AWAIT_WADD_AND_DATA;
        end
        else case (writeState)
            AWAIT_WADD_AND_DATA: begin
                case ({s_axi_lite_awvalid, s_axi_lite_wvalid})
                    2'b10: begin writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WDATA; end
                    2'b01: begin writeData <= s_axi_lite_wdata; writeState <= AWAIT_WADD; end
                    2'b11: begin writeData <= s_axi_lite_wdata; writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WRITE; end
                    default: writeState <= AWAIT_WADD_AND_DATA;
                endcase
            end
            AWAIT_WDATA: begin if (s_axi_lite_wvalid) begin writeData <= s_axi_lite_wdata; writeState <= AWAIT_WRITE; end end
            AWAIT_WADD:  begin if (s_axi_lite_awvalid) begin writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WRITE; end end
            AWAIT_WRITE: begin regfile[writeAddr] <= writeData; writeState <= AWAIT_RESP; end
            AWAIT_RESP:  begin if (s_axi_lite_bready) writeState <= AWAIT_WADD_AND_DATA; end
            default: writeState <= AWAIT_WADD_AND_DATA;
        endcase
    end

    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready  = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid  = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp   = (writeAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;

    // Display raster counter
    reg [9:0] x;
    reg [8:0] y;
    reg       first_start=0;

    wire first = (x == 0) && (y == 0);
    wire lastx = (x == X_SIZE - 1);
    wire lasty = (y == Y_SIZE - 1);
    wire ready;  // driven by packer.in_stream_ready (packer output port)

    // Lookahead counter - feeds input_builder, advances on l1_advance_lx_out
    reg [9:0] lx;
    reg [8:0] ly;
    wire      llastx = (lx == X_SIZE - 1);
    wire      llasty = (ly == Y_SIZE - 1);

    // Input builder
    wire signed [15:0] input_vec [N_FEATURES-1:0];

    input_builder #(.N_FEATURES(N_FEATURES)) ib (
        .clk(out_stream_aclk),
        .resetn(periph_resetn),
        .pixel_x(lx), .pixel_y(ly),
        .axis_x_select(axis_x_select),
        .axis_y_select(axis_y_select),
        .feature_regs(feature_regs),
        .x(input_vec)
    );

    // MLP pipeline
    // wr_bank encoding: 0=L1, 1=L2 (all 4 banks decode internally), 2=L3
    wire [7:0]  mlp_r, mlp_g, mlp_b;
    wire        mlp_valid;
    reg         mlp_start;
    wire        l1_advance_lx_out;

    // Derive wr_bank from upper bits of weight_addr register
    // Convention: software writes bits [17:16] of the weight_addr register
    // to select the target layer: 0=L1, 1=L2, 2=L3
    wire [1:0] wr_bank = weight_addr[15:14];

    mlp_pipeline #(
        .N_FEATURES(N_FEATURES),
        .H1_SIZE(H1_SIZE),
        .H2_SIZE(H2_SIZE)
    ) mlp (
        .clk(out_stream_aclk),
        .resetn(periph_resetn),
        .start(mlp_start),
        .x(input_vec),
        .wr_addr(weight_addr),
        .wr_data(weight_data),
        .wr_en(weight_we),
        .wr_bank(wr_bank),
        .z_offset(z_offset),
        .z_scale(z_scale),
        .z_shift(z_shift),
        .r(mlp_r), .g(mlp_g), .b(mlp_b),
        .valid(mlp_valid),
        .l1_advance_lx_out(l1_advance_lx_out)
    );

    // Pixel FIFO - absorbs MLP output during downstream backpressure
    localparam FIFO_DEPTH = 128;
    localparam FIFO_ABITS = 7;

    // packed as {r[7:0], g[7:0], b[7:0], eol, sof} = 26 bits
    reg [25:0]           fifo_mem [FIFO_DEPTH-1:0];
    reg [FIFO_ABITS-1:0] fifo_wr_ptr;
    reg [FIFO_ABITS-1:0] fifo_rd_ptr;
    reg [FIFO_ABITS:0]   fifo_count;

    wire fifo_empty = (fifo_count == 0);
    wire fifo_full  = (fifo_count == FIFO_DEPTH);
    wire fifo_push  = mlp_valid && !fifo_full;
    wire fifo_pop   = !fifo_empty && ready;  // ready = packer.in_stream_ready

    wire [25:0] fifo_rd_data = fifo_mem[fifo_rd_ptr];

    // Raster counter control
    always @(posedge out_stream_aclk) begin
        mlp_start <= 0;

        if (!periph_resetn) begin
            x           <= 0; y           <= 0;
            lx          <= 0; ly          <= 0;
            first_start <= 0;
            fifo_wr_ptr <= 0; fifo_rd_ptr <= 0; fifo_count <= 0;
        end
        else begin
            if (!first_start) begin
                mlp_start   <= 1;
                first_start <= 1;
            end

            // advance lookahead when L1 is about to restart
            if (l1_advance_lx_out) begin
                if (llastx) begin lx <= 10'd0; ly <= llasty ? 9'd0 : ly + 9'd1; end
                else lx <= lx + 10'd1;
            end

            // push pixel into FIFO when MLP produces; capture eol/sof from
            // current x/y then advance the display counter
            if (fifo_push) begin
                fifo_mem[fifo_wr_ptr] <= {mlp_r, mlp_g, mlp_b, lastx, first};
                fifo_wr_ptr <= fifo_wr_ptr + 1;
                if (lastx) begin x <= 10'd0; y <= lasty ? 9'd0 : y + 9'd1; end
                else x <= x + 10'd1;
            end

            // pop when packer consumes
            if (fifo_pop)
                fifo_rd_ptr <= fifo_rd_ptr + 1;

            // update count (simultaneous push+pop leaves count unchanged)
            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1;
                2'b01: fifo_count <= fifo_count - 1;
                default: ;
            endcase
        end
    end

    // Pixel packer - reads from FIFO and produces AXI-Stream output, applying backpressure to the FIFO when needed
    packer pixel_packer (
        .aclk(out_stream_aclk), .aresetn(periph_resetn),
        .r(fifo_rd_data[25:18]), .g(fifo_rd_data[17:10]), .b(fifo_rd_data[9:2]),
        .eol(fifo_rd_data[1]), .in_stream_ready(ready),
        .valid(!fifo_empty), .sof(fifo_rd_data[0]),
        .out_stream_tdata(out_stream_tdata),
        .out_stream_tkeep(out_stream_tkeep),
        .out_stream_tlast(out_stream_tlast),
        .out_stream_tready(out_stream_tready),
        .out_stream_tvalid(out_stream_tvalid),
        .out_stream_tuser(out_stream_tuser)
    );

    assign dbg_r     = DEBUG_PORTS ? mlp_r     : 8'h0;
    assign dbg_g     = DEBUG_PORTS ? mlp_g     : 8'h0;
    assign dbg_b     = DEBUG_PORTS ? mlp_b     : 8'h0;
    assign dbg_valid = DEBUG_PORTS ? mlp_valid : 1'b0;
    assign dbg_x     = DEBUG_PORTS ? x         : 10'h0;
    assign dbg_y     = DEBUG_PORTS ? y         : 9'h0;

endmodule
