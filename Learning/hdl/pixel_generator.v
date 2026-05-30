module pixel_generator #(
    parameter AXI_LITE_ADDR_WIDTH = 8,
    parameter REG_FILE_SIZE = 75
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
    input           s_axi_lite_wvalid
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

    // -------------------------------------------------------
    // Register file
    // -------------------------------------------------------
    reg [31:0] regfile [REG_FILE_SIZE-1:0];
    reg [REG_FILE_AWIDTH-1:0] writeAddr, readAddr;
    reg [31:0]                readData, writeData;
    reg [1:0]                 readState  = AWAIT_RADD;
    reg [2:0]                 writeState = AWAIT_WADD_AND_DATA;

    // extract named signals from regfile
    wire signed [15:0] feature_regs [N_FEATURES-1:0];
    genvar fi;
    generate
        for (fi = 0; fi < N_FEATURES; fi = fi + 1) begin
            assign feature_regs[fi] = regfile[fi][15:0];
        end
    endgenerate

    wire [5:0]  axis_x_select = regfile[64][5:0];
    wire [5:0]  axis_y_select = regfile[65][5:0];
    wire [5:0]  n_features    = regfile[66][5:0];
    wire [5:0]  h1_size       = regfile[67][5:0];
    wire [5:0]  h2_size       = regfile[68][5:0];
    wire [15:0] weight_addr   = regfile[69][15:0];
    wire [7:0]  weight_data   = regfile[70][7:0];
    wire        weight_we     = regfile[71][0];
    wire signed [31:0] z_offset = regfile[72];
    wire [31:0] z_scale       = regfile[73];

    // -------------------------------------------------------
    // AXI-Lite read state machine
    // -------------------------------------------------------
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
            AWAIT_FETCH: begin
                readState <= AWAIT_READ;
            end
            AWAIT_READ: begin
                if (s_axi_lite_rready)
                    readState <= AWAIT_RADD;
            end
            default: readState <= AWAIT_RADD;
        endcase
    end

    assign s_axi_lite_arready = (readState == AWAIT_RADD);
    assign s_axi_lite_rresp   = (readAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;
    assign s_axi_lite_rvalid  = (readState == AWAIT_READ);
    assign s_axi_lite_rdata   = readData;

    // -------------------------------------------------------
    // AXI-Lite write state machine
    // -------------------------------------------------------
    always @(posedge s_axi_lite_aclk) begin
        if (!axi_resetn) begin
            writeState <= AWAIT_WADD_AND_DATA;
        end
        else case (writeState)
            AWAIT_WADD_AND_DATA: begin
                case ({s_axi_lite_awvalid, s_axi_lite_wvalid})
                    2'b10: begin
                        writeAddr  <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                        writeState <= AWAIT_WDATA;
                    end
                    2'b01: begin
                        writeData  <= s_axi_lite_wdata;
                        writeState <= AWAIT_WADD;
                    end
                    2'b11: begin
                        writeData  <= s_axi_lite_wdata;
                        writeAddr  <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                        writeState <= AWAIT_WRITE;
                    end
                    default: writeState <= AWAIT_WADD_AND_DATA;
                endcase
            end
            AWAIT_WDATA: begin
                if (s_axi_lite_wvalid) begin
                    writeData  <= s_axi_lite_wdata;
                    writeState <= AWAIT_WRITE;
                end
            end
            AWAIT_WADD: begin
                if (s_axi_lite_awvalid) begin
                    writeAddr  <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                    writeState <= AWAIT_WRITE;
                end
            end
            AWAIT_WRITE: begin
                regfile[writeAddr] <= writeData;
                writeState         <= AWAIT_RESP;
            end
            AWAIT_RESP: begin
                if (s_axi_lite_bready)
                    writeState <= AWAIT_WADD_AND_DATA;
            end
            default: writeState <= AWAIT_WADD_AND_DATA;
        endcase
    end

    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready  = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid  = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp   = (writeAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;

    // -------------------------------------------------------
    // Display raster counter (tracks outputs already sent to packer)
    // -------------------------------------------------------
    reg [9:0] x;
    reg [8:0] y;
    reg       first_start;

    wire first = (x == 0) && (y == 0);
    wire lastx = (x == X_SIZE - 1);
    wire lasty = (y == Y_SIZE - 1);
    wire ready = out_stream_tready;

    // -------------------------------------------------------
    // Lookahead counter (one pixel ahead — feeds input_builder)
    // Advances when L1 is about to restart so input_vec is ready in time.
    // -------------------------------------------------------
    reg [9:0] lx;
    reg [8:0] ly;
    wire      llastx = (lx == X_SIZE - 1);
    wire      llasty = (ly == Y_SIZE - 1);

    // -------------------------------------------------------
    // Input builder — uses lookahead coordinates
    // -------------------------------------------------------
    wire signed [15:0] input_vec [N_FEATURES-1:0];

    input_builder #(
        .N_FEATURES(N_FEATURES)
    ) ib (
        .clk(out_stream_aclk),
        .resetn(periph_resetn),
        .pixel_x(lx),
        .pixel_y(ly),
        .axis_x_select(axis_x_select),
        .axis_y_select(axis_y_select),
        .feature_regs(feature_regs),
        .x(input_vec)
    );

    // -------------------------------------------------------
    // MLP pipeline
    // -------------------------------------------------------
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
        .r(mlp_r),
        .g(mlp_g),
        .b(mlp_b),
        .valid(mlp_valid),
        .l1_advance_lx_out(l1_advance_lx_out)
    );

    // -------------------------------------------------------
    // Raster counter control - gated by mlp_valid
    // -------------------------------------------------------
    wire valid_int = mlp_valid;

    always @(posedge out_stream_aclk) begin
        mlp_start <= 0;

        if (!periph_resetn) begin
            x           <= 0;
            y           <= 0;
            lx          <= 0;
            ly          <= 0;
            mlp_start   <= 0;
            first_start <= 0;
        end
        else begin
            // one-shot start — subsequent pixels triggered by l1_early_trigger inside mlp_pipeline
            if (!first_start) begin
                mlp_start   <= 1;
                first_start <= 1;
            end

            // advance lookahead when L1 is about to restart (one cycle before l1_stall resolves)
            // input_builder sees updated lx/ly on the stall cycle → output ready when L1 reads x[0]
            if (l1_advance_lx_out) begin
                if (llastx) begin
                    lx <= 10'd0;
                    if (llasty) ly <= 9'd0;
                    else        ly <= ly + 9'd1;
                end
                else lx <= lx + 10'd1;
            end

            // display counter follows pipeline outputs
            if (mlp_valid && ready) begin
                if (lastx) begin
                    x <= 10'd0;
                    if (lasty) y <= 9'd0;
                    else       y <= y + 9'd1;
                end
                else x <= x + 10'd1;
            end
        end
    end

    // -------------------------------------------------------
    // Pixel packer
    // -------------------------------------------------------
    packer pixel_packer (
        .aclk(out_stream_aclk),
        .aresetn(periph_resetn),
        .r(mlp_r), .g(mlp_g), .b(mlp_b),
        .eol(lastx),
        .in_stream_ready(ready),
        .valid(valid_int),
        .sof(first),
        .out_stream_tdata(out_stream_tdata),
        .out_stream_tkeep(out_stream_tkeep),
        .out_stream_tlast(out_stream_tlast),
        .out_stream_tready(out_stream_tready),
        .out_stream_tvalid(out_stream_tvalid),
        .out_stream_tuser(out_stream_tuser)
    );

    reg [31:0] cycle_count = 0;
    always @(posedge out_stream_aclk) begin
        cycle_count <= cycle_count + 1;

        // print every 100 cycles
       if (cycle_count % 100 == 0)
        $display("Cycle %0d: mlp_start=%b mlp_valid=%b first_start=%b x=%0d y=%0d",
            cycle_count, mlp_start, mlp_valid, first_start, x, y);

        // print when mlp_start fires
        if (mlp_start)
            $display("Cycle %0d: mlp_start fired", cycle_count);

        // print when mlp_valid fires
        if (mlp_valid)
            $display("Cycle %0d: mlp_valid fired! RGB=%0d,%0d,%0d", cycle_count, mlp_r, mlp_g, mlp_b);
    end

endmodule