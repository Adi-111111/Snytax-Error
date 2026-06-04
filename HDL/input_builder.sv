module input_builder #(
    parameter int N_FEATURES = 16
)(
    input  logic        clk,
    input  logic        resetn,
    input  logic [9:0]  pixel_x,
    input  logic [8:0]  pixel_y,
    input  logic [5:0]  axis_x_select,
    input  logic [5:0]  axis_y_select,
    input  logic signed [15:0] feature_regs [N_FEATURES-1:0],
    output logic signed [15:0] x            [N_FEATURES-1:0]
);

    // 1/639 in Q1.15: round(32768/639) = 51
    // 1/479 in Q1.15: round(32768/479) = 68
    localparam logic [15:0] INV_639 = 16'd51;
    localparam logic [15:0] INV_479 = 16'd68;

    logic [25:0] x_norm_full;
    logic [25:0] y_norm_full;
    logic signed [15:0] x_norm;
    logic signed [15:0] y_norm;

    always_comb begin
        x_norm_full = pixel_x * INV_639;
        y_norm_full = pixel_y * INV_479;
        x_norm      = x_norm_full[15:0];
        y_norm      = y_norm_full[15:0];
    end

    always_ff @(posedge clk) begin
        if (!resetn) begin
            for (int i = 0; i < N_FEATURES; i++)
                x[i] <= '0;
        end else begin
            for (int i = 0; i < N_FEATURES; i++) begin
                if (i == axis_x_select)
                    x[i] <= x_norm;
                else if (i == axis_y_select)
                    x[i] <= y_norm;
                else
                    x[i] <= feature_regs[i];
            end
        end
    end

endmodule
