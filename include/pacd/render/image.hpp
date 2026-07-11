#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "pacd/render/math.hpp"

namespace pacd::render {

// 8-bit RGB raster, row-major with the origin at the top-left corner.
//
// PNG encoding is self-contained (no libpng/zlib): pixels are emitted through
// "stored" deflate blocks so the output is a valid, if uncompressed, PNG that
// any viewer can open. That keeps the render library dependency-free.
class Image {
   public:
    Image(int width, int height);
    Image(int width, int height, Vec3 fill_color);

    [[nodiscard]] int width() const noexcept { return m_width; }
    [[nodiscard]] int height() const noexcept { return m_height; }

    // Colour channels are in [0, 1] and are clamped; out-of-bounds writes are
    // silently ignored so rasterizer edge cases cannot corrupt memory.
    void set_pixel(int x, int y, Vec3 color) noexcept;
    [[nodiscard]] Vec3 pixel(int x, int y) const noexcept;

    [[nodiscard]] std::vector<std::uint8_t> to_png() const;
    [[nodiscard]] bool save_png(std::string_view path) const;

   private:
    [[nodiscard]] bool in_bounds(int x, int y) const noexcept;
    [[nodiscard]] std::size_t offset(int x, int y) const noexcept;

    int m_width;
    int m_height;
    std::vector<std::uint8_t> m_pixels;  // width * height * 3 bytes
};

}  // namespace pacd::render
