#include <chrono>
#include <iostream>
#include <functional>

#include <nuklear.h>
#include <lightvis/lightvis.h>
#include <lightvis/shader.h>
#include <glbinding-aux/types_to_string.h>

static unsigned int msb(unsigned int x) {
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return (x & ~(x >> 1));
}

struct Image {
    Image(const cv::Mat &image) {
        gl::glGenTextures(1, &texture_id);
        gl::glBindTexture(gl::GL_TEXTURE_2D, texture_id);
        gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MIN_FILTER, gl::GL_LINEAR_MIPMAP_LINEAR);
        gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_MAG_FILTER, gl::GL_LINEAR);
        gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_S, gl::GL_CLAMP_TO_EDGE);
        gl::glTexParameteri(gl::GL_TEXTURE_2D, gl::GL_TEXTURE_WRAP_T, gl::GL_CLAMP_TO_EDGE);
        gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
        nuklear_image = nk_image_id((int)texture_id);

        update_image(image);
    }

    ~Image() {
        gl::glDeleteTextures(1, &texture_id);
    }

    bool empty() const {
        return (size.x() == 0 || size.y() == 0);
    }

    void update_image(const cv::Mat &image) {
        size.x() = image.cols;
        size.y() = image.rows;

        if (empty()) return;

        texture_size.x() = std::min((int)msb((unsigned int)size.x()), 2048);
        texture_size.y() = std::min((int)msb((unsigned int)size.y()), 2048);

        cv::Mat rgb;
        cv::resize(image, rgb, cv::Size(texture_size.x(), texture_size.y()));
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        gl::glBindTexture(gl::GL_TEXTURE_2D, texture_id);
        gl::glTexImage2D(gl::GL_TEXTURE_2D, 0, gl::GL_RGB, texture_size.x(), texture_size.y(), 0, gl::GL_RGB, gl::GL_UNSIGNED_BYTE, rgb.ptr());
        gl::glGenerateMipmap(gl::GL_TEXTURE_2D);
        gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
    }

    struct nk_image nuklear_image;
    gl::GLuint texture_id;
    Eigen::Vector2i texture_size;
    Eigen::Vector2i size;
};

class PlayerVisualizer : public lightvis::LightVis {

    int is_playing = 0;

    std::vector<Eigen::Vector3f> trajectory;
    Eigen::Vector4f trajectory_color;

    cv::Mat feature_tracker_cvimage;
    std::unique_ptr<Image> feature_tracker_image;

  public:
    PlayerVisualizer(bool play) :
        LightVis("Player", 1200, 720) {
        trajectory_color = {1.0, 0.25, 0.4, 1.0};
        add_trajectory(trajectory, trajectory_color);
        trajectory.push_back(Eigen::Vector3f(0.0, 0.0, 1.0));
        // add_points(landmarks, landmarks_color);

        is_playing = (int)play;
    }

    void load() override {
        feature_tracker_image = std::make_unique<Image>(feature_tracker_cvimage);
    }

    void unload() override {
        feature_tracker_image.reset();
    }

    bool step() {
        trajectory.push_back(trajectory.back() + Eigen::Vector3f(0.1, 0.1, 0.0));
        return true;
    }

    void gui(void *ctx, int w, int h) override {
        feature_tracker_image->update_image(feature_tracker_cvimage);
        auto *context = (nk_context *)(ctx);
        context->style.window.spacing = nk_vec2(0, 0);
        context->style.window.padding = nk_vec2(0, 0);
        context->style.window.border = 1.0;
        context->style.window.fixed_background = nk_style_item_color(nk_rgba(48, 48, 48, 128));

        if (nk_begin(context, "Controls", nk_rect(0, h - 40, 240, 40), NK_WINDOW_NO_SCROLLBAR)) {
            nk_layout_row_static(context, 40, 80, 3);
            if (is_playing) {
                is_playing = step();
            }
            if (is_playing) {
                if (nk_button_label(context, "Playing")) {
                    is_playing = false;
                }
            } else {
                if (nk_button_label(context, "Stopped")) {
                    is_playing = true;
                }
                nk_button_push_behavior(context, NK_BUTTON_REPEATER);
                if (nk_button_label(context, "Forward")) {
                    step();
                }
                nk_button_pop_behavior(context);
                nk_button_push_behavior(context, NK_BUTTON_DEFAULT);
                if (nk_button_label(context, "Step")) {
                    step();
                }
                nk_button_pop_behavior(context);
            }
        }
        nk_end(context);

        const int tw = 240;

        if (!feature_tracker_image->empty()) {
            struct nk_rect rect;
            rect.x = 0;
            rect.y = 0;
            rect.w = tw;
            rect.h = ((float)tw / feature_tracker_image->size.x()) * feature_tracker_image->size.y();
            if (nk_begin(context, "Displays", nk_rect(0, 0, rect.w, rect.h + 10), NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_BORDER)) {
                //auto rect = nk_window_get_content_region(context);
                nk_layout_space_begin(context, NK_STATIC, rect.h, INT_MAX);
                nk_layout_space_push(context, rect);
                nk_image(context, feature_tracker_image->nuklear_image);
                nk_layout_space_end(context);
                double a = 0;
                a = std::min(a, 2.0);
                nk_command_buffer *canvas = nk_window_get_canvas(context);
                if (a < 0.2) {
                    nk_fill_rect(canvas, nk_rect(0, rect.h, tw, 10), 0, nk_rgb(255, 0, 0));
                }
                nk_fill_rect(canvas, nk_rect(0, rect.h, a / 2.0 * tw, 10), 0, nk_rgb(255, 255, 0));
            }
            nk_end(context);
        }

        context->style.window.fixed_background = nk_style_item_color(nk_rgba(32, 32, 32, 0));

        nk_begin(context, "Overlays", nk_rect(w - 100, 0, 100, 30), NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NO_INPUT);

            char fps_text[14];
            snprintf(fps_text, 14, "FPS: % 8.3f", 30.0);
            nk_command_buffer *canvas = nk_window_get_canvas(context);
            nk_draw_text(canvas, nk_rect(w - 90, 5, 90, 25), fps_text, 13, context->style.font, nk_rgba(255, 255, 255, 0), nk_rgba(255, 255, 255, 255));

        nk_end(context);
    }

    void draw(int w, int h) override {
        auto err = gl::glGetError();
        if (err != gl::GL_NONE) {
            std::cout << err << std::endl;
            exit(0);
        }
    }
};

int main(int argc, char *argv[]) {
    PlayerVisualizer vis(false);
    vis.show();
    return lightvis::main();
}
