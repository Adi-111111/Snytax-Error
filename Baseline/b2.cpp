#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <SDL2/SDL.h>

const int   N_FEATURES   = 16;
const int   H1           = 64;
const int   H2           = 32;
const int   N_CLASSES    = 7;
const int   WIDTH        = 640;
const int   HEIGHT       = 480;
const int   PANEL        = 120;
const char* WEIGHTS_FILE = "weights.hex";

// Evenly spaced hues — must match Python make_colours(7)
const uint8_t CLASS_R[N_CLASSES] = {255,  255,   72,    0,    0,   72,  255};
const uint8_t CLASS_G[N_CLASSES] = {  0,  218,  255,  255,  145,    0,    0};
const uint8_t CLASS_B[N_CLASSES] = {  0,    0,    0,  145,  255,  255,  218};

float relu(float x) { return x > 0.0f ? x : 0.0f; }
int   clamp_int(int x, int lo, int hi) { return std::max(lo, std::min(hi, x)); }
float clamp_float(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }

std::vector<float> read_q44_line(std::ifstream& f) {
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || (line.size() >= 2 && line[0] == '/' && line[1] == '/')) continue;
        std::vector<float> v;
        std::istringstream ss(line);
        std::string tok;
        while (ss >> tok) {
            int8_t b = (int8_t)(uint8_t)std::stoul(tok, nullptr, 16);
            v.push_back((float)b / 16.0f);
        }
        return v;
    }
    return {};
}

float read_q1616_signed(std::ifstream& f) {
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || (line.size() >= 2 && line[0] == '/' && line[1] == '/')) continue;
        return (float)(int32_t)(uint32_t)std::stoul(line, nullptr, 16) / 65536.0f;
    }
    return 0.0f;
}

float read_q816_unsigned(std::ifstream& f) {
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || (line.size() >= 2 && line[0] == '/' && line[1] == '/')) continue;
        return (float)(uint32_t)std::stoul(line, nullptr, 16) / 65536.0f;
    }
    return 0.0f;
}

void forward_layer(
    const std::vector<float>& W, const std::vector<float>& b,
    const std::vector<float>& x, std::vector<float>& y,
    int in_size, int out_size, bool use_relu)
{
    for (int j = 0; j < out_size; j++) {
        float sum = b[j];
        for (int i = 0; i < in_size; i++) sum += W[j * in_size + i] * x[i];
        y[j] = use_relu ? relu(sum) : sum;
    }
}

void logits_to_rgb(const float z[3], const float z_offset[3], const float z_scale[3],
                   int& r, int& g, int& b)
{
    float norm[3];
    for (int k = 0; k < 3; k++)
        norm[k] = clamp_float((z[k] - z_offset[k]) * z_scale[k] / 255.0f, 0.0f, 1.0f);

    r = clamp_int((int)std::round(norm[0] * 255.0f), 0, 255);
    g = clamp_int((int)std::round(norm[1] * 255.0f), 0, 255);
    b = clamp_int((int)std::round(norm[2] * 255.0f), 0, 255);
}

void draw_slider(SDL_Renderer* ren, int idx, float val, bool sel, bool axis) {
    int x0 = 80, y0 = HEIGHT + 10 + idx * 6, w = 500, h = 4;
    if (axis)     SDL_SetRenderDrawColor(ren, 70,  70,  70,  255);
    else if (sel) SDL_SetRenderDrawColor(ren, 230, 230, 230, 255);
    else          SDL_SetRenderDrawColor(ren, 140, 140, 140, 255);
    SDL_Rect bar = {x0, y0, w, h};
    SDL_RenderFillRect(ren, &bar);
    if (axis) SDL_SetRenderDrawColor(ren, 90, 90, 90, 255);
    else      SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    SDL_Rect knob = {x0 + (int)(val * w) - 3, y0 - 4, 6, 12};
    SDL_RenderFillRect(ren, &knob);
}

void render(
    SDL_Renderer* renderer, SDL_Texture* texture,
    const std::vector<float>& W1, const std::vector<float>& b1,
    const std::vector<float>& W2, const std::vector<float>& b2,
    const std::vector<float>& W3, const std::vector<float>& b3,
    const float z_offset[3], const float z_scale[3],
    int axis_x, int axis_y, int sel,
    const std::vector<float>& bg)
{
    std::vector<uint32_t> pixels(WIDTH * HEIGHT);
    std::vector<float> x(N_FEATURES), a1(H1), a2(H2), z(3);

    for (int py = 0; py < HEIGHT; py++) {
        for (int px = 0; px < WIDTH; px++) {
            x = bg;
            x[axis_x] = (float)px / (float)(WIDTH  - 1);
            x[axis_y] = (float)py / (float)(HEIGHT - 1);

            forward_layer(W1, b1, x,  a1, N_FEATURES, H1, true);
            forward_layer(W2, b2, a1, a2, H1,         H2, true);
            forward_layer(W3, b3, a2, z,  H2,         3,  false);

            int r, g, b;
            logits_to_rgb(z.data(), z_offset, z_scale, r, g, b);

            pixels[py * WIDTH + px] =
                (255u << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
        }
    }

    SDL_UpdateTexture(texture, nullptr, pixels.data(), WIDTH * sizeof(uint32_t));
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);

    SDL_SetRenderDrawColor(renderer, 25, 25, 25, 255);
    SDL_Rect panel = {0, HEIGHT, WIDTH, PANEL};
    SDL_RenderFillRect(renderer, &panel);

    for (int i = 0; i < N_FEATURES; i++)
        draw_slider(renderer, i, bg[i], i == sel, i == axis_x || i == axis_y);

    SDL_RenderPresent(renderer);
}

void smooth(std::vector<float>& cur, const std::vector<float>& tgt) {
    for (int i = 0; i < N_FEATURES; i++) {
        float d = tgt[i] - cur[i];
        cur[i] += std::fabs(d) > 0.0005f ? d * 0.08f : d;
    }
}

int main() {
    std::ifstream file(WEIGHTS_FILE);
    if (!file.is_open()) { std::cerr << "Cannot open " << WEIGHTS_FILE << "\n"; return 1; }

    float z_offset[3], z_scale[3];
    for (int k = 0; k < 3; k++) {
        z_offset[k] = read_q1616_signed(file);
        z_scale[k]  = read_q816_unsigned(file);
    }

    std::vector<float> W1 = read_q44_line(file), b1 = read_q44_line(file);
    std::vector<float> W2 = read_q44_line(file), b2 = read_q44_line(file);
    std::vector<float> W3 = read_q44_line(file), b3 = read_q44_line(file);
    file.close();

    for (int k = 0; k < 3; k++)
        std::cout << "Channel " << k << ": offset=" << z_offset[k] << "  scale=" << z_scale[k] << "\n";
    std::cout << "W1=" << W1.size() << "  W2=" << W2.size() << "  W3=" << W3.size() << "\n";

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window*   win = SDL_CreateWindow("MLP Decision Surface",
                            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                            WIDTH, HEIGHT + PANEL, 0);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture*  tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
                            SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    int axis_x = 0, axis_y = 4, sel = 2;
    std::vector<float> cur_bg(N_FEATURES, 0.5f), tgt_bg(N_FEATURES, 0.5f);

    bool running = true;
    int frames = 0;
    auto fps_start = std::chrono::high_resolution_clock::now();

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_KEYDOWN) {
                SDL_Keycode k = ev.key.keysym.sym;
                if (k == SDLK_ESCAPE) running = false;
                if (k == SDLK_x) { axis_x = (axis_x+1)%N_FEATURES; if (axis_x==axis_y) axis_x=(axis_x+1)%N_FEATURES; }
                if (k == SDLK_y) { axis_y = (axis_y+1)%N_FEATURES; if (axis_y==axis_x) axis_y=(axis_y+1)%N_FEATURES; }
                if (k == SDLK_TAB) sel = (sel+1)%N_FEATURES;
                if (k == SDLK_LEFT  && sel!=axis_x && sel!=axis_y) tgt_bg[sel] = clamp_float(tgt_bg[sel]-0.10f, 0.0f, 1.0f);
                if (k == SDLK_RIGHT && sel!=axis_x && sel!=axis_y) tgt_bg[sel] = clamp_float(tgt_bg[sel]+0.10f, 0.0f, 1.0f);
                if (k == SDLK_r) std::fill(tgt_bg.begin(), tgt_bg.end(), 0.5f);
            }
        }

        smooth(cur_bg, tgt_bg);
        render(ren, tex, W1, b1, W2, b2, W3, b3, z_offset, z_scale, axis_x, axis_y, sel, cur_bg);

        frames++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            int fps = (int)(frames / elapsed);
            frames = 0;
            fps_start = now;
            SDL_SetWindowTitle(win, ("MLP Decision Surface | FPS: " + std::to_string(fps/2) +
                " | X=f" + std::to_string(axis_x) + " Y=f" + std::to_string(axis_y) +
                " sel=f" + std::to_string(sel)).c_str());
        }
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
