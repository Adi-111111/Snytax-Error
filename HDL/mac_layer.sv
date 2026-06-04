module mac_layer #(
    parameter int N_INPUTS   = 16,
    parameter int N_NEURONS  = 64,
    parameter int APPLY_RELU = 1,
    parameter int IN_WIDTH   = 16,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                        clk,
    input  logic                        resetn,
    input  logic                        start,
    input  logic signed [IN_WIDTH-1:0]  in_val,
    input  logic [(N_NEURONS*8)-1:0]    weights_flat,
    input  logic [7:0]                  n_inputs,
    output logic signed [ACC_WIDTH-1:0] out [N_NEURONS-1:0],
    output logic                        done
);
    logic signed [7:0] w [N_NEURONS-1:0];
    generate
        for (genvar i = 0; i < N_NEURONS; i++) begin : unpack
            assign w[i] = weights_flat[i*8 +: 8];
        end
    endgenerate

    logic signed [ACC_WIDTH-1:0] accumulator [N_NEURONS-1:0];
    logic signed [ACC_WIDTH-1:0] final_sum   [N_NEURONS-1:0];
    logic [7:0]                  count;
    logic                        running;
    logic                        latch_out;

    generate
        for (genvar k = 0; k < N_NEURONS; k++) begin : final_sum_gen
            assign final_sum[k] = accumulator[k] + ACC_WIDTH'(in_val) * ACC_WIDTH'(w[k]);
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!resetn) begin
            done      <= '0;
            latch_out <= '0;
            running   <= '0;
            count     <= '0;
            for (int j = 0; j < N_NEURONS; j++)
                accumulator[j] <= '0;
        end else begin
            done      <= latch_out;
            latch_out <= '0;

            if (start) begin
                running <= 1'b1;
                count   <= 8'd1;
                for (int j = 0; j < N_NEURONS; j++)
                    accumulator[j] <= ACC_WIDTH'(in_val) * ACC_WIDTH'(w[j]);
            end else if (running) begin
                for (int j = 0; j < N_NEURONS; j++)
                    accumulator[j] <= accumulator[j] + ACC_WIDTH'(in_val) * ACC_WIDTH'(w[j]);
                count <= count + 8'd1;

                if (count == n_inputs - 1) begin
                    running   <= 1'b0;
                    latch_out <= 1'b1;
                end
            end
        end
    end

    // Output register - explicit hold on every cycle forces Vivado to use
    // the D pin for all cases including the zero (ReLU) case, never the R pin.
    // This breaks the DSP-cascade -> synchronous-reset critical path.
    always_ff @(posedge clk) begin
        for (int j = 0; j < N_NEURONS; j++) begin
            if (latch_out) begin
                if (APPLY_RELU && final_sum[j][ACC_WIDTH-1])
                    out[j] <= '0;
                else
                    out[j] <= final_sum[j];
            end else begin
                out[j] <= out[j];
            end
        end
    end

endmodule
