module packer( 

    input aclk, 
    input aresetn, 

    input [7:0] r,g,b, 
    input eol, 
    output in_stream_ready, //backpressure signal going back to pixel_gen
    input valid, 
    input sof, //start of frame

    output [31:0]   out_stream_tdata, 
    output [3:0]    out_stream_tkeep, 
    output          out_stream_tlast, 
    input           out_stream_tready, 
    output          out_stream_tvalid, 
    output [0:0]    out_stream_tuser   
); 

reg [1:0] state_reg = 2'b0; 

wire [1:0] state = sof ? 2'b00 : state_reg; 
wire state0 = (state == 2'b0); 

reg sof_reg; 
reg [7:0]   last_r, last_g, last_b; 

//combinational 
reg[31:0]   tdata; 
reg         tvalid; 
reg         ready;

always @(posedge aclk) begin
    if(aresetn) begin
        //advance state if valid and... 
        if(valid) begin 
            //...if in state 0 or destination is ready
            if(state0 || out_stream_tready) begin 
                if (eol) begin //stay synchronised to start of new line - so reset to state 0 on eol
                    state_reg <= 2'b0;
                end 
                else begin 
                    state_reg <= state + 2'b1; 
                end 

                last_r <= r; 
                last_g <= g; 
                last_b <= b; 
            end 

            //store the sof flag when it is set (it can't be read in this cycle because the data isn't ready)
            if (sof) begin
                sof_reg <= 1'b1; 
            end
            //reset it after it has been read by the VDMA 
            else if (valid & out_stream_tready) begin
                sof_reg <= 1'b0;
            end 
        end 
    end
    else begin 
        state_reg <= 2'b0; 
        sof_reg <= 1'b0; 
    end 
end 

always @* begin 
    case ({state}) 
        2'b00: begin 
            //output is not complete (valid) in this state --> means we are always ready for the next pixel
            tdata = {g, last_r, last_b, last_g}; //don't care since valid is false - just copy another state
            tvalid = 1'b0; 
            ready = 1'b1;
        end 
        2'b01: begin 
            tdata = {g, last_r, last_b, last_g}; 
            tvalid = valid;
            ready = out_stream_tready; 
        end 
        2'b10: begin 
            tdata = {b, g, last_r, last_b}; 
            tvalid = valid; 
            ready = out_stream_tready; 
        end 
        2'b11: begin 
            tdata = {r,b,g, last_r}; 
            tvalid = valid; 
            ready = out_stream_tready; 
        end 
        default: begin 
            tdata = {g, last_r, last_b, last_g}; 
            tvalid = 1'b0; 
            ready = 1'b1; 
        end 
    endcase
end 

assign in_stream_ready = ready; 
assign out_stream_tlast = eol; //eol goes high on the last pixel of each row 
assign out_stream_tuser = sof_reg; //signal to vdma that there is a start of frame 
assign out_stream_tkeep = 4'hf; //all 4 bytes are valid - assume a line length is a multiple of 4 bytes
assign out_stream_tdata = tdata; 
assign out_stream_tvalid = tvalid; 

endmodule 

