module input_builder #(
    parameter N_FEATURES = 16
)(
    input                           clk,
    input                           resetn,

    // pixel coordinates from raster counter
    input [9:0]                     pixel_x,
    input [8:0]                     pixel_y,

    // axis selection
    input [5:0]                     axis_x_select,
    input [5:0]                     axis_y_select,

    // feature registers from AXI - all Q1.15
    input signed [15:0]             feature_regs [N_FEATURES-1:0],

    // output vector - one value per feature
    output reg signed [15:0]        x [N_FEATURES-1:0]
);

    // precomputed reciprocals in Q1.15
    // 1/639 × 32768 = 51.28 → 51
    // 1/479 × 32768 = 68.41 → 68
    localparam [15:0] INV_639 = 16'd51;
    localparam [15:0] INV_479 = 16'd68;

    // normalised pixel coordinates - full width to avoid truncation
    wire signed [25:0] x_norm_full = pixel_x * INV_639;
    wire signed [25:0] y_norm_full = pixel_y * INV_479;

    // take bottom 16 bits - correct Q1.15 result
    wire signed [15:0] x_norm = x_norm_full[15:0];
    wire signed [15:0] y_norm = y_norm_full[15:0];

    integer i;

    always @(posedge clk) begin
        if (!resetn) begin
            for (i = 0; i < N_FEATURES; i = i + 1)
                x[i] <= 0;
        end
        else begin
            for (i = 0; i < N_FEATURES; i = i + 1) begin
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