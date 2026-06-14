module pixel_generator #(
    parameter AXI_LITE_ADDR_WIDTH = 9,
    parameter REG_FILE_SIZE = 86,
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

    // debug outputs — only used in testbench (DEBUG_PORTS=1)
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
        regfile[75] = 32'd0; // display_enable: 0 while loading, 1 when ready
        regfile[76] = 32'd1; // overlay_enable
        regfile[77] = 32'd320; // crosshair_x
        regfile[78] = 32'd240; // crosshair_y
        regfile[79] = 32'd320; // y_axis_x position in pixels
        regfile[80] = 32'd240; // x_axis_y position in pixels
        regfile[81] = 32'd1; // overlay thickness in pixels
        regfile[82] = 32'h00FFFFFF; // crosshair colour RGB
        regfile[83] = 32'h00FFFFFF; // axis colour RGB
        regfile[84] = 32'd20; // crosshair half-size in pixels
        regfile[85] = 32'd0; // reserved
    end

    reg [31:0] live_regfile [REG_FILE_SIZE-1:0];

    wire signed [15:0] feature_regs [N_FEATURES-1:0];
    genvar fi;
    generate
        for (fi = 0; fi < N_FEATURES; fi = fi + 1) begin
            assign feature_regs[fi] = live_regfile[fi][15:0];
        end
    endgenerate

    wire [5:0]  axis_x_select = live_regfile[64][5:0];
    wire [5:0]  axis_y_select = live_regfile[65][5:0];

    // Weight writes are commands and must use the AXI-Lite shadow regfile.
    wire [15:0] weight_addr   = regfile[69][15:0];
    wire [7:0]  weight_data   = regfile[70][7:0];
    wire        weight_we     = regfile[71][0];

    wire signed [31:0] z_offset = live_regfile[72];
    wire [31:0] z_scale         = live_regfile[73];
    wire [4:0]  z_shift         = live_regfile[74][4:0];
    wire        display_enable  = live_regfile[75][0];

    wire        overlay_enable = live_regfile[76][0];
    wire [9:0]  crosshair_x    = (live_regfile[77][9:0] >= X_SIZE) ? 10'd639 : live_regfile[77][9:0];
    wire [8:0]  crosshair_y    = (live_regfile[78][8:0] >= Y_SIZE) ? 9'd479 : live_regfile[78][8:0];
    wire [9:0]  axis_x_pos     = (live_regfile[79][9:0] >= X_SIZE) ? 10'd639 : live_regfile[79][9:0];
    wire [8:0]  axis_y_pos     = (live_regfile[80][8:0] >= Y_SIZE) ? 9'd479 : live_regfile[80][8:0];
    wire [3:0]  overlay_raw_thickness = live_regfile[81][3:0];
    wire [3:0]  overlay_thickness = (overlay_raw_thickness == 4'd0) ? 4'd1 : overlay_raw_thickness;
    wire [23:0] crosshair_colour = live_regfile[82][23:0];
    wire [23:0] axis_colour      = live_regfile[83][23:0];
    wire [9:0]  crosshair_half_size = (live_regfile[84][9:0] == 10'd0) ? 10'd20 : live_regfile[84][9:0];

    always @(posedge s_axi_lite_aclk) begin
        readData <= (readAddr < REG_FILE_SIZE) ? regfile[readAddr] : 32'd0;
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
            AWAIT_WRITE: begin
                if (writeAddr < REG_FILE_SIZE)
                    regfile[writeAddr] <= writeData;
                writeState <= AWAIT_RESP;
            end
            AWAIT_RESP:  begin if (s_axi_lite_bready) writeState <= AWAIT_WADD_AND_DATA; end
            default: writeState <= AWAIT_WADD_AND_DATA;
        endcase
    end

    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready  = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid  = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp   = (writeAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;

    reg [9:0] x;
    reg [8:0] y;
    reg       first_start=0;

    wire first = (x == 0) && (y == 0);
    wire lastx = (x == X_SIZE - 1);
    wire lasty = (y == Y_SIZE - 1);
    wire ready;  // driven by packer.in_stream_ready (packer output port)

    integer li;
    always @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            for (li = 0; li < REG_FILE_SIZE; li = li + 1)
                live_regfile[li] <= 32'd0;

            for (li = 0; li < 16; li = li + 1)
                live_regfile[li] <= 32'd16384;

            live_regfile[64] <= 32'd0;
            live_regfile[65] <= 32'd1;
            live_regfile[66] <= 32'd16;
            live_regfile[67] <= 32'd64;
            live_regfile[68] <= 32'd32;
            live_regfile[71] <= 32'd0;
            live_regfile[72] <= 32'd0;
            live_regfile[73] <= 32'd65536;
            live_regfile[74] <= 32'd16;
            live_regfile[75] <= 32'd0;
            live_regfile[76] <= 32'd1;
            live_regfile[77] <= 32'd320;
            live_regfile[78] <= 32'd240;
            live_regfile[79] <= 32'd320;
            live_regfile[80] <= 32'd240;
            live_regfile[81] <= 32'd1;
            live_regfile[82] <= 32'h00FFFFFF;
            live_regfile[83] <= 32'h00FFFFFF;
            live_regfile[84] <= 32'd20;
            live_regfile[85] <= 32'd0;
        end else if (first) begin
            for (li = 0; li < REG_FILE_SIZE; li = li + 1)
                live_regfile[li] <= regfile[li];
        end
    end

    reg [9:0] lx;
    reg [8:0] ly;
    wire      llastx = (lx == X_SIZE - 1);
    wire      llasty = (ly == Y_SIZE - 1);

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

    wire [7:0]  mlp_r, mlp_g, mlp_b;
    wire        mlp_valid;
    reg         mlp_start;
    wire        l1_advance_lx_out;

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

    // Pixel FIFO — absorbs MLP output during downstream backpressure
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


    wire [10:0] dx_cross = (x >= crosshair_x) ? {1'b0, x - crosshair_x} : {1'b0, crosshair_x - x};
    wire [9:0]  dy_cross = (y >= crosshair_y) ? {1'b0, y - crosshair_y} : {1'b0, crosshair_y - y};
    wire [10:0] dx_axis  = (x >= axis_x_pos)  ? {1'b0, x - axis_x_pos}  : {1'b0, axis_x_pos - x};
    wire [9:0]  dy_axis  = (y >= axis_y_pos)  ? {1'b0, y - axis_y_pos}  : {1'b0, axis_y_pos - y};

    wire on_cross_v = overlay_enable && (dx_cross < overlay_thickness) && (dy_cross <= crosshair_half_size[9:0]);
    wire on_cross_h = overlay_enable && (dy_cross < overlay_thickness) && (dx_cross <= {1'b0, crosshair_half_size});
    wire on_cross   = on_cross_v || on_cross_h;

    wire on_axis_v  = overlay_enable && (dx_axis < overlay_thickness);
    wire on_axis_h  = overlay_enable && (dy_axis < overlay_thickness);
    wire on_axis    = on_axis_v || on_axis_h;

    wire [7:0] base_r = display_enable ? mlp_r : 8'd0;
    wire [7:0] base_g = display_enable ? mlp_g : 8'd0;
    wire [7:0] base_b = display_enable ? mlp_b : 8'd0;

    wire [7:0] out_r = on_cross ? crosshair_colour[23:16] : (on_axis ? axis_colour[23:16] : base_r);
    wire [7:0] out_g = on_cross ? crosshair_colour[15:8]  : (on_axis ? axis_colour[15:8]  : base_g);
    wire [7:0] out_b = on_cross ? crosshair_colour[7:0]   : (on_axis ? axis_colour[7:0]   : base_b);

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
                fifo_mem[fifo_wr_ptr] <= {
                    out_r,
                    out_g,
                    out_b,
                    lastx,
                    first
                };
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

    // Pixel packer — reads from FIFO and produces AXI-Stream output, applying backpressure to the FIFO when needed
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