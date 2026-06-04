// mac_layer.v
// Parameterised MAC layer with optional parallel input processing.
//
// N_PARALLEL = 1  -> original single-input behaviour (L1)
// N_PARALLEL = 4  -> 4 inputs processed per cycle (L2, 64/4=16 cycles)
// N_PARALLEL = 2  -> 2 inputs processed per cycle (L3, 32/2=16 cycles)
//
// n_inputs is the FULL input count — the module divides by N_PARALLEL
// internally. So for L2 pass n_inputs=H1_SIZE=64, not H1_SIZE/4.
//
// Counter starts at N_PARALLEL on the start cycle (first group already
// processed), increments by N_PARALLEL each running cycle, fires done
// when count >= n_inputs.

module mac_layer #(
    parameter N_INPUTS    = 16,
    parameter N_NEURONS   = 64,
    parameter APPLY_RELU  = 1,
    parameter IN_WIDTH    = 16,
    parameter ACC_WIDTH   = 32,
    parameter N_PARALLEL  = 1
)(
    input                                       clk,
    input                                       resetn,
    input                                       start,

    // flattened input bus: N_PARALLEL inputs packed LSB-first
    // in_val_flat[p*IN_WIDTH +: IN_WIDTH] = input p
    input  signed [(N_PARALLEL*IN_WIDTH)-1:0]   in_val_flat,

    // flattened weight bus: N_PARALLEL weight columns packed
    // weights_flat[p*N_NEURONS*8 + i*8 +: 8] = weight for parallel p, neuron i
    input  [(N_PARALLEL*N_NEURONS*8)-1:0]       weights_flat,

    // full input count (mac divides by N_PARALLEL internally); 7 bits holds up to 128
    input  [6:0]                                n_inputs,

    output reg [(N_NEURONS*ACC_WIDTH)-1:0]      out_flat,
    output reg                                  done
);

    reg signed [ACC_WIDTH-1:0]  accumulator [N_NEURONS-1:0];
    reg signed [ACC_WIDTH-1:0]  final_mac   [N_NEURONS-1:0];
    reg [6:0]                   count;       // counts processed inputs (0..N_INPUTS)
    reg                         running;
    integer                     i;

    // unpack individual weights and inputs via generate
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

    // parallel sum width: product is IN_WIDTH+8 bits, sum of N_PARALLEL needs
    // log2(N_PARALLEL) extra bits; +3 handles up to N_PARALLEL=8
    localparam PAR_SUM_W = IN_WIDTH + 8 + 3;

    // combinatorial parallel sum per neuron
    wire signed [PAR_SUM_W-1:0] par_sum [N_NEURONS-1:0];
    generate
        for (gi = 0; gi < N_NEURONS; gi = gi + 1) begin : gen_psum
            wire signed [PAR_SUM_W-1:0] ps [N_PARALLEL:0];
            assign ps[0] = 0;
            for (gp = 0; gp < N_PARALLEL; gp = gp + 1) begin : gen_add
                wire signed [IN_WIDTH+7:0] prod = iv[gp] * w[gp][gi];
                assign ps[gp+1] = ps[gp] + {{(PAR_SUM_W-IN_WIDTH-8){prod[IN_WIDTH+7]}}, prod};
            end
            assign par_sum[gi] = ps[N_PARALLEL];
        end
    endgenerate
    
   // After the par_sum generate block, add a registered stage
    reg signed [PAR_SUM_W-1:0] par_sum_reg [N_NEURONS-1:0];
    integer pi;
    always @(posedge clk) begin
        for (pi = 0; pi < N_NEURONS; pi = pi + 1)
            par_sum_reg[pi] <= par_sum[pi];
    end

    always @(posedge clk) begin
        if (!resetn) begin
            done    <= 0;
            running <= 0;
            count   <= 0;
            for (i = 0; i < N_NEURONS; i = i + 1)
                accumulator[i] <= 0;
        end

        else begin
            done <= 0;

            if (start) begin
                running <= 1;
                count   <= N_PARALLEL[6:0];
                for (i = 0; i < N_NEURONS; i = i + 1)
                    accumulator[i] <= par_sum_reg[i];
            end

            else if (running) begin
                if (count + N_PARALLEL[6:0] >= n_inputs) begin
                    running <= 0;
                    done    <= 1;
                    for (i = 0; i < N_NEURONS; i = i + 1) begin
                        final_mac[i] = accumulator[i] + par_sum_reg[i];
                        if (APPLY_RELU && final_mac[i][ACC_WIDTH-1])
                            out_flat[i*ACC_WIDTH +: ACC_WIDTH] <= 0;
                        else
                            out_flat[i*ACC_WIDTH +: ACC_WIDTH] <= final_mac[i];
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
