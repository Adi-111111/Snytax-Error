module pixel_generator #(
    parameter int AXI_LITE_ADDR_WIDTH = 9,
    parameter int REG_FILE_SIZE       = 82
)(
    input  logic        out_stream_aclk,
    input  logic        s_axi_lite_aclk,
    input  logic        axi_resetn,
    input  logic        periph_resetn,

    output logic [31:0] out_stream_tdata,
    output logic [3:0]  out_stream_tkeep,
    output logic        out_stream_tlast,
    input  logic        out_stream_tready,
    output logic        out_stream_tvalid,
    output logic [0:0]  out_stream_tuser,

    input  logic [AXI_LITE_ADDR_WIDTH-1:0] s_axi_lite_araddr,
    output logic        s_axi_lite_arready,
    input  logic        s_axi_lite_arvalid,
    input  logic [AXI_LITE_ADDR_WIDTH-1:0] s_axi_lite_awaddr,
    output logic        s_axi_lite_awready,
    input  logic        s_axi_lite_awvalid,
    input  logic        s_axi_lite_bready,
    output logic [1:0]  s_axi_lite_bresp,
    output logic        s_axi_lite_bvalid,
    output logic [31:0] s_axi_lite_rdata,
    input  logic        s_axi_lite_rready,
    output logic [1:0]  s_axi_lite_rresp,
    output logic        s_axi_lite_rvalid,
    input  logic [31:0] s_axi_lite_wdata,
    output logic        s_axi_lite_wready,
    input  logic        s_axi_lite_wvalid
);

    localparam int X_SIZE          = 640;
    localparam int Y_SIZE          = 480;
    localparam int N_FEATURES      = 16;
    localparam int H1_SIZE         = 64;
    localparam int H2_SIZE         = 32;
    localparam int REG_FILE_AWIDTH = 7;

    localparam logic [2:0] AWAIT_WADD_AND_DATA = 3'b000;
    localparam logic [2:0] AWAIT_WDATA         = 3'b001;
    localparam logic [2:0] AWAIT_WADD          = 3'b010;
    localparam logic [2:0] AWAIT_WRITE         = 3'b100;
    localparam logic [2:0] AWAIT_RESP          = 3'b101;
    localparam logic [1:0] AWAIT_RADD          = 2'b00;
    localparam logic [1:0] AWAIT_FETCH         = 2'b01;
    localparam logic [1:0] AWAIT_READ          = 2'b10;
    localparam logic [1:0] AXI_OK              = 2'b00;
    localparam logic [1:0] AXI_ERR             = 2'b10;

    // shadow: ARM always writes here
    // regfile: updated atomically from shadow at VSYNC
    logic [31:0] shadow  [REG_FILE_SIZE-1:0];
    logic [31:0] regfile [REG_FILE_SIZE-1:0];

    (* max_fanout = 8 *) logic [REG_FILE_AWIDTH-1:0] writeAddr, readAddr;
    logic [31:0] readData, writeData;
    logic [1:0]  readState  = AWAIT_RADD;
    logic [2:0]  writeState = AWAIT_WADD_AND_DATA;

    // feature registers 0..15
    logic signed [15:0] feature_regs [N_FEATURES-1:0];
    generate
        for (genvar fi = 0; fi < N_FEATURES; fi++) begin
            assign feature_regs[fi] = regfile[fi][15:0];
        end
    endgenerate

    logic [5:0]  live_axis_x;     assign live_axis_x     = regfile[64][5:0];
    logic [5:0]  live_axis_y;     assign live_axis_y     = regfile[65][5:0];
    logic [5:0]  live_n_features; assign live_n_features = regfile[66][5:0];
    logic [5:0]  live_h1_size;    assign live_h1_size    = regfile[67][5:0];
    logic [5:0]  live_h2_size;    assign live_h2_size    = regfile[68][5:0];

    // weight write registers - read from shadow so writes take effect immediately
    logic [15:0] weight_addr; assign weight_addr = shadow[69][15:0];
    logic [7:0]  weight_data; assign weight_data = shadow[70][7:0];
    logic        weight_we;   assign weight_we   = shadow[71][0];

    // per-channel normalisation
    logic signed [31:0] z_offset [2:0];
    logic        [31:0] z_scale  [2:0];
    assign z_offset[0] = regfile[72]; assign z_scale[0] = regfile[73];
    assign z_offset[1] = regfile[74]; assign z_scale[1] = regfile[75];
    assign z_offset[2] = regfile[76]; assign z_scale[2] = regfile[77];

    logic [9:0] live_dot_x; assign live_dot_x = regfile[78][9:0];
    logic [8:0] live_dot_y; assign live_dot_y = regfile[79][8:0];

    // rendering halt reads shadow directly so it takes effect immediately,
    // not at the next VSYNC - needed to pause rendering before weight writes
    logic rendering_halt; assign rendering_halt = shadow[80][0];

    logic vsync;
    assign vsync = (x == '0) && (y == '0);

    always_ff @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            for (int i = 0; i < REG_FILE_SIZE; i++)
                regfile[i] <= '0;
        end else if (vsync) begin
            for (int i = 0; i < REG_FILE_SIZE; i++)
                regfile[i] <= shadow[i];
        end
    end

    always_ff @(posedge s_axi_lite_aclk) begin
        readData <= regfile[readAddr];

        if (!axi_resetn) begin
            readState <= AWAIT_RADD;
        end else begin
            case (readState)
                AWAIT_RADD:  if (s_axi_lite_arvalid) begin
                                 readAddr  <= s_axi_lite_araddr[2+:REG_FILE_AWIDTH];
                                 readState <= AWAIT_FETCH;
                             end
                AWAIT_FETCH: readState <= AWAIT_READ;
                AWAIT_READ:  if (s_axi_lite_rready) readState <= AWAIT_RADD;
                default:     readState <= AWAIT_RADD;
            endcase
        end
    end

    assign s_axi_lite_arready = (readState == AWAIT_RADD);
    assign s_axi_lite_rresp   = (readAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;
    assign s_axi_lite_rvalid  = (readState == AWAIT_READ);
    assign s_axi_lite_rdata   = readData;

    always_ff @(posedge s_axi_lite_aclk) begin
        if (!axi_resetn) begin
            writeState <= AWAIT_WADD_AND_DATA;
        end else begin
            case (writeState)
                AWAIT_WADD_AND_DATA: begin
                    case ({s_axi_lite_awvalid, s_axi_lite_wvalid})
                        2'b10: begin writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WDATA;  end
                        2'b01: begin writeData <= s_axi_lite_wdata;                      writeState <= AWAIT_WADD;   end
                        2'b11: begin writeData <= s_axi_lite_wdata;
                                     writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WRITE;  end
                        default: writeState <= AWAIT_WADD_AND_DATA;
                    endcase
                end
                AWAIT_WDATA: if (s_axi_lite_wvalid)  begin writeData <= s_axi_lite_wdata;                      writeState <= AWAIT_WRITE; end
                AWAIT_WADD:  if (s_axi_lite_awvalid) begin writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WRITE; end
                AWAIT_WRITE: begin
                    // reject writes that would make axis_x == axis_y
                    if (!((writeAddr == 7'd65 && writeData[5:0] == shadow[64][5:0]) ||
                          (writeAddr == 7'd64 && writeData[5:0] == shadow[65][5:0])))
                        shadow[writeAddr] <= writeData;
                    writeState <= AWAIT_RESP;
                end
                AWAIT_RESP: if (s_axi_lite_bready) writeState <= AWAIT_WADD_AND_DATA;
                default:    writeState <= AWAIT_WADD_AND_DATA;
            endcase
        end
    end

    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready  = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid  = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp   = (writeAddr  < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;

    logic [9:0] x,  lx;
    logic [8:0] y,  ly;
    logic       first_start;

    wire first  = (x == '0) && (y == '0);
    wire lastx  = (x == X_SIZE - 1);
    wire lasty  = (y == Y_SIZE - 1);
    wire llastx = (lx == X_SIZE - 1);
    wire llasty = (ly == Y_SIZE - 1);

    logic packer_ready;

    logic signed [15:0] input_vec [N_FEATURES-1:0];

    input_builder #(.N_FEATURES(N_FEATURES)) ib (
        .clk(out_stream_aclk),
        .resetn(periph_resetn),
        .pixel_x(lx),
        .pixel_y(ly),
        .axis_x_select(live_axis_x),
        .axis_y_select(live_axis_y),
        .feature_regs(feature_regs),
        .x(input_vec)
    );

    logic [7:0] mlp_r, mlp_g, mlp_b;
    logic       mlp_valid;
    logic       mlp_start;
    logic       l1_advance_lx_out;

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
        .wr_en(weight_we && rendering_halt),
        .wr_bank(weight_addr[15:14]),
        .z_offset(z_offset),
        .z_scale(z_scale),
        .r(mlp_r), .g(mlp_g), .b(mlp_b),
        .valid(mlp_valid),
        .l1_advance_lx_out(l1_advance_lx_out)
    );

    // freeze display during rendering halt - pipeline still runs but output discarded
    wire valid_int = mlp_valid && !rendering_halt;

    always_ff @(posedge out_stream_aclk) begin
        mlp_start <= 1'b0;

        if (!periph_resetn) begin
            x           <= '0;
            y           <= '0;
            lx          <= '0;
            ly          <= '0;
            first_start <= 1'b0;
        end else begin
            // one-shot pulse to kick off the pipeline at reset release
            if (!first_start) begin
                mlp_start   <= 1'b1;
                first_start <= 1'b1;
            end

            // advance lookahead when L1 is about to restart
            if (l1_advance_lx_out) begin
                if (llastx) begin
                    lx <= '0;
                    ly <= llasty ? '0 : ly + 9'(1);
                end else begin
                    lx <= lx + 10'(1);
                end
            end

            // display counter advances with each valid pixel accepted
            if (mlp_valid && out_stream_tready) begin
                if (lastx) begin
                    x <= '0;
                    y <= lasty ? '0 : y + 9'(1);
                end else begin
                    x <= x + 10'(1);
                end
            end
        end
    end

    logic [7:0] pix_r, pix_g, pix_b;
    logic       on_crosshair;

    always_comb begin
        on_crosshair = (x == live_dot_x) || (y == live_dot_y);
        pix_r = on_crosshair ? 8'hFF : mlp_r;
        pix_g = on_crosshair ? 8'hFF : mlp_g;
        pix_b = on_crosshair ? 8'hFF : mlp_b;
    end

    packer pixel_packer (
        .aclk(out_stream_aclk),
        .aresetn(periph_resetn),
        .r(pix_r), .g(pix_g), .b(pix_b),
        .eol(lastx),
        .in_stream_ready(packer_ready),
        .valid(valid_int),
        .sof(first),
        .out_stream_tdata(out_stream_tdata),
        .out_stream_tkeep(out_stream_tkeep),
        .out_stream_tlast(out_stream_tlast),
        .tready(out_stream_tready),
        .out_stream_tvalid(out_stream_tvalid),
        .out_stream_tuser(out_stream_tuser)
    );

endmodule