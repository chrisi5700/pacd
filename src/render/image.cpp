#include "pacd/render/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "pacd/render/math.hpp"

namespace pacd::render {

namespace {

constexpr int CHANNELS = 3;
constexpr std::uint16_t MAX_STORED_BLOCK = 0xFFFF;
constexpr std::uint32_t ADLER_MODULUS = 65521;

[[nodiscard]] std::uint8_t to_byte(float value) noexcept {
    const float clamped = std::clamp(value, 0.0F, 1.0F);
    return static_cast<std::uint8_t>(std::lround(clamped * 255.0F));
}

void push_be32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

[[nodiscard]] std::uint32_t crc32(const std::vector<std::uint8_t>& data) noexcept {
    constexpr std::uint32_t POLYNOMIAL = 0xEDB88320U;
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const std::uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const bool low_set = (crc & 1U) != 0U;
            crc >>= 1U;
            if (low_set) {
                crc ^= POLYNOMIAL;
            }
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

[[nodiscard]] std::uint32_t adler32(const std::vector<std::uint8_t>& data) noexcept {
    std::uint32_t low = 1;
    std::uint32_t high = 0;
    for (const std::uint8_t byte : data) {
        low = (low + byte) % ADLER_MODULUS;
        high = (high + low) % ADLER_MODULUS;
    }
    return (high << 16U) | low;
}

void write_chunk(std::vector<std::uint8_t>& out, std::string_view type,
                 const std::vector<std::uint8_t>& data) {
    push_be32(out, static_cast<std::uint32_t>(data.size()));
    std::vector<std::uint8_t> typed;
    typed.reserve(type.size() + data.size());
    std::ranges::transform(type, std::back_inserter(typed),
                           [](char character) { return static_cast<std::uint8_t>(character); });
    typed.insert(typed.end(), data.begin(), data.end());
    out.insert(out.end(), typed.begin(), typed.end());
    push_be32(out, crc32(typed));
}

// Wrap the raw (filtered) scanline stream in a zlib container using only
// uncompressed "stored" deflate blocks.
[[nodiscard]] std::vector<std::uint8_t> zlib_store(const std::vector<std::uint8_t>& raw) {
    std::vector<std::uint8_t> out;
    out.push_back(0x78);  // zlib CMF: deflate, 32K window
    out.push_back(0x01);  // zlib FLG: no preset dict, check bits
    std::size_t pos = 0;
    while (pos < raw.size()) {
        const std::size_t remaining = raw.size() - pos;
        const std::size_t count = std::min<std::size_t>(remaining, MAX_STORED_BLOCK);
        const bool last = (pos + count) == raw.size();
        out.push_back(last ? std::uint8_t{1} : std::uint8_t{0});
        const auto len = static_cast<std::uint16_t>(count);
        const auto nlen = static_cast<std::uint16_t>(~len);
        out.push_back(static_cast<std::uint8_t>(len & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((len >> 8U) & 0xFFU));
        out.push_back(static_cast<std::uint8_t>(nlen & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((nlen >> 8U) & 0xFFU));
        out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
                   raw.begin() + static_cast<std::ptrdiff_t>(pos + count));
        pos += count;
    }
    push_be32(out, adler32(raw));
    return out;
}

}  // namespace

Image::Image(int width, int height) : Image(width, height, Vec3{0.0F, 0.0F, 0.0F}) {}

Image::Image(int width, int height, Vec3 fill_color)
    : m_width(std::max(width, 0)), m_height(std::max(height, 0)) {
    const auto pixel_count =
        static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height);
    m_pixels.resize(pixel_count * CHANNELS);
    const std::array<std::uint8_t, CHANNELS> rgb{to_byte(fill_color.x), to_byte(fill_color.y),
                                                 to_byte(fill_color.z)};
    for (std::size_t index = 0; index < pixel_count; ++index) {
        for (std::size_t channel = 0; channel < CHANNELS; ++channel) {
            m_pixels.at((index * CHANNELS) + channel) = rgb.at(channel);
        }
    }
}

bool Image::in_bounds(int x, int y) const noexcept {
    return x >= 0 && y >= 0 && x < m_width && y < m_height;
}

std::size_t Image::offset(int x, int y) const noexcept {
    const auto row = static_cast<std::size_t>(y) * static_cast<std::size_t>(m_width);
    return (row + static_cast<std::size_t>(x)) * CHANNELS;
}

void Image::set_pixel(int x, int y, Vec3 color) noexcept {
    if (!in_bounds(x, y)) {
        return;
    }
    const std::size_t base = offset(x, y);
    m_pixels.at(base + 0) = to_byte(color.x);
    m_pixels.at(base + 1) = to_byte(color.y);
    m_pixels.at(base + 2) = to_byte(color.z);
}

Vec3 Image::pixel(int x, int y) const noexcept {
    if (!in_bounds(x, y)) {
        return {0.0F, 0.0F, 0.0F};
    }
    const std::size_t base = offset(x, y);
    constexpr float INV = 1.0F / 255.0F;
    return {static_cast<float>(m_pixels.at(base + 0)) * INV,
            static_cast<float>(m_pixels.at(base + 1)) * INV,
            static_cast<float>(m_pixels.at(base + 2)) * INV};
}

std::vector<std::uint8_t> Image::to_png() const {
    // Build the filtered scanline stream: each row is prefixed with filter 0.
    std::vector<std::uint8_t> raw;
    const auto stride = static_cast<std::size_t>(m_width) * CHANNELS;
    raw.reserve((stride + 1) * static_cast<std::size_t>(m_height));
    for (int row = 0; row < m_height; ++row) {
        raw.push_back(0);  // filter type: None
        const std::size_t row_base = static_cast<std::size_t>(row) * stride;
        for (std::size_t byte = 0; byte < stride; ++byte) {
            raw.push_back(m_pixels.at(row_base + byte));
        }
    }

    std::vector<std::uint8_t> ihdr;
    push_be32(ihdr, static_cast<std::uint32_t>(m_width));
    push_be32(ihdr, static_cast<std::uint32_t>(m_height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // colour type: truecolour RGB
    ihdr.push_back(0);  // compression: deflate
    ihdr.push_back(0);  // filter method: adaptive
    ihdr.push_back(0);  // interlace: none

    std::vector<std::uint8_t> png{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    write_chunk(png, "IHDR", ihdr);
    write_chunk(png, "IDAT", zlib_store(raw));
    write_chunk(png, "IEND", {});
    return png;
}

bool Image::save_png(std::string_view path) const {
    std::ofstream file(std::string{path}, std::ios::binary);
    if (!file) {
        return false;
    }
    const std::vector<std::uint8_t> bytes = to_png();
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

}  // namespace pacd::render
