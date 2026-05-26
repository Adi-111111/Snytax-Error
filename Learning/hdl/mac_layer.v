module mac_layer #(
    parameter N_INPUTS   = 16,
    parameter N_NEURONS  = 64,
    parameter APPLY_RELU = 1,
    parameter IN_WIDTH   = 16,
    parameter ACC_WIDTH  = 32
)(
    input                           clk,
    input                           resetn,
    input                           start,
    input signed [IN_WIDTH-1:0]     in_val,
    input [(N_NEURONS*8)-1:0]       weights_flat,
    input        [7:0]              n_inputs,
    output reg signed [ACC_WIDTH-1:0] out [N_NEURONS-1:0],
    output reg                      done
);

    reg signed [ACC_WIDTH-1:0] accumulator [N_NEURONS-1:0];
    reg signed [ACC_WIDTH-1:0] final_mac   [N_NEURONS-1:0];
    reg [5:0]                  count;
    reg                        running;
    integer                    j;

    // slice individual weights out of flat bus
    wire signed [7:0] w [N_NEURONS-1:0];
    genvar i;
    generate
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            assign w[i] = weights_flat[i*8 +: 8];
        end
    endgenerate

    always @(posedge clk) begin
        if (!resetn) begin
            done    <= 0;
            running <= 0;
            count   <= 0;
            for (j = 0; j < N_NEURONS; j = j + 1)
                accumulator[j] <= 0;
        end

        else begin
            done <= 0;

            if (start) begin
                running <= 1;
                count   <= 1;
                for (j = 0; j < N_NEURONS; j = j + 1)
                    accumulator[j] <= in_val * w[j];
            end

            else if (running) begin
                for (j = 0; j < N_NEURONS; j = j + 1)
                    accumulator[j] <= accumulator[j] + in_val * w[j];

                count <= count + 1;

                if (count == n_inputs - 1) begin
                    running <= 0;
                    done    <= 1;

                    for (j = 0; j < N_NEURONS; j = j + 1) begin
                        final_mac[j] = accumulator[j] + in_val * w[j];
                        if (APPLY_RELU && final_mac[j][ACC_WIDTH-1])
                            out[j] <= 0;
                        else
                            out[j] <= final_mac[j];
                    end
                end
            end
        end
    end

endmodule