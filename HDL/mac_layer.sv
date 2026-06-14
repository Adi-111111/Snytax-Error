
module mac_layer #(
    parameter N_INPUTS    = 16,
    parameter N_NEURONS   = 64,
    parameter APPLY_RELU  = 1,
    parameter IN_WIDTH    = 16,
    parameter ACC_WIDTH   = 32,
    parameter N_PARALLEL  = 1,
    parameter USE_DSP     = 1
)(
    input                                       clk,
    input                                       resetn,
    input                                       start,
    input  signed [(N_PARALLEL*IN_WIDTH)-1:0]   in_val_flat,
    input  [(N_PARALLEL*N_NEURONS*8)-1:0]       weights_flat,
    input  [6:0]                                n_inputs,

    output reg signed [ACC_WIDTH-1:0]           out [N_NEURONS-1:0],
    output reg                                  done
);

    reg signed [ACC_WIDTH-1:0]  accumulator   [N_NEURONS-1:0];
    reg signed [ACC_WIDTH-1:0]  final_mac     [N_NEURONS-1:0];
    reg [6:0]                   count;
    reg                         running;
    integer                     i;

    wire signed [7:0]          w  [N_PARALLEL-1:0][N_NEURONS-1:0];
    wire signed [IN_WIDTH-1:0] iv [N_PARALLEL-1:0];

    genvar gp, gi;
    generate
        for (gp = 0; gp < N_PARALLEL; gp = gp + 1) begin : gen_unpack_p
            assign iv[gp] = in_val_flat[gp*IN_WIDTH +: IN_WIDTH];
            for (gi = 0; gi < N_NEURONS; gi = gi + 1) begin : gen_unpack_i
                assign w[gp][gi] = weights_flat[(gp*N_NEURONS + gi)*8 +: 8];
            end
        end
    endgenerate

    localparam PAR_SUM_W = IN_WIDTH + 8 + 3;

    reg signed [PAR_SUM_W-1:0]  par_sum_reg [N_NEURONS-1:0];
    wire signed [PAR_SUM_W-1:0] par_sum [N_NEURONS-1:0];
    generate
        for (gi = 0; gi < N_NEURONS; gi = gi + 1) begin : gen_psum
            wire signed [PAR_SUM_W-1:0] ps [N_PARALLEL:0];
            assign ps[0] = 0;
            for (gp = 0; gp < N_PARALLEL; gp = gp + 1) begin : gen_add
                if (USE_DSP != 0) begin : gen_prod_dsp
                    (* use_dsp = "yes" *) wire signed [IN_WIDTH+7:0] prod = iv[gp] * w[gp][gi];
                    assign ps[gp+1] = ps[gp] + {{(PAR_SUM_W-IN_WIDTH-8){prod[IN_WIDTH+7]}}, prod};
                end else begin : gen_prod_lut
                    (* use_dsp = "no" *) wire signed [IN_WIDTH+7:0] prod = iv[gp] * w[gp][gi];
                    assign ps[gp+1] = ps[gp] + {{(PAR_SUM_W-IN_WIDTH-8){prod[IN_WIDTH+7]}}, prod};
                end
            end
            assign par_sum[gi] = ps[N_PARALLEL];
        end
    endgenerate

    always @(posedge clk) begin
        if (!resetn) begin
            done    <= 0;
            running <= 0;
            count   <= 0;
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                accumulator[i]  <= 0;
                par_sum_reg[i]  <= 0;
            end
        end

        else begin
            done <= 0;

            for (i = 0; i < N_NEURONS; i = i + 1)
                par_sum_reg[i] <= par_sum[i];

            if (start) begin
                running <= 1;
                count   <= 0;  
            end

            else if (running) begin
                if (count == 0) begin
                    for (i = 0; i < N_NEURONS; i = i + 1)
                        accumulator[i] <= par_sum_reg[i];
                    count <= N_PARALLEL[6:0];
                end
                else if (count + N_PARALLEL[6:0] >= n_inputs) begin
                    running <= 0;
                    done    <= 1;
                    for (i = 0; i < N_NEURONS; i = i + 1) begin
                        final_mac[i] = accumulator[i] + par_sum_reg[i];
                        if (APPLY_RELU && final_mac[i][ACC_WIDTH-1])
                            out[i] <= 0;
                        else
                            out[i] <= final_mac[i];
                    end
                end
                else begin
                    count <= count + N_PARALLEL[6:0];
                    for (i = 0; i < N_NEURONS; i = i + 1)
                        accumulator[i] <= accumulator[i] + par_sum_reg[i];
                end
            end
        end
    end

endmodule
