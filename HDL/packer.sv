module packer (
    input  logic        aclk,
    input  logic        aresetn,
    input  logic [7:0]  r, g, b,
    input  logic        eol,
    output logic        in_stream_ready,
    input  logic        valid,
    input  logic        sof,
    output logic [31:0] out_stream_tdata,
    output logic [3:0]  out_stream_tkeep,
    output logic        out_stream_tlast,
    input  logic        tready,
    output logic        out_stream_tvalid,
    output logic [0:0]  out_stream_tuser
);

    logic [1:0] state_reg = 2'b0;
    wire [1:0]  state  = sof ? 2'b00 : state_reg;
    wire        state0 = (state == 2'b0);

    logic        sof_reg;
    logic [7:0]  last_r, last_g, last_b;
    logic [31:0] tdata;
    logic        tvalid;
    logic        ready;

    always_ff @(posedge aclk) begin
        if (aresetn) begin
            if (valid) begin
                if (state0 | tready) begin
                    if (eol)
                        state_reg <= 2'b0;
                    else
                        state_reg <= state + 2'b1;
                    last_r <= r;
                    last_g <= g;
                    last_b <= b;
                end
                if (sof)
                    sof_reg <= 1'b1;
                else if (valid & tready)
                    sof_reg <= 1'b0;
            end
        end else begin
            state_reg <= 2'b0;
            sof_reg   <= 1'b0;
        end
    end

    always_comb begin
        case (state)
            2'b00: begin
                tdata  = {g, last_r, last_b, last_g};
                tvalid = 1'b0;
                ready  = 1'b1;
            end
            2'b01: begin
                tdata  = {g, last_r, last_b, last_g};
                tvalid = valid;
                ready  = tready;
            end
            2'b10: begin
                tdata  = {b, g, last_r, last_b};
                tvalid = valid;
                ready  = tready;
            end
            2'b11: begin
                tdata  = {r, b, g, last_r};
                tvalid = valid;
                ready  = tready;
            end
            default: begin
                tdata  = {g, last_r, last_b, last_g};
                tvalid = 1'b0;
                ready  = 1'b1;
            end
        endcase
    end

    assign in_stream_ready   = ready;
    assign out_stream_tlast  = eol;
    assign out_stream_tuser  = sof_reg;
    assign out_stream_tkeep  = 4'hf;
    assign out_stream_tdata  = tdata;
    assign out_stream_tvalid = tvalid;

endmodule
