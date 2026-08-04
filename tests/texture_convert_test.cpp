// BC decompression, for devices whose GPU cannot sample BC at all — which is
// every Android GPU, BC being a desktop format. Getting these wrong does not
// fail loudly: a texture pack simply looks subtly incorrect, and only on the
// platforms that need this path most.
//
// The interpolation cases exist because aurora already has an S3TCBlend()
// helper that is NOT the S3TC spec: it is the GameCube's (3a+5b)/8
// approximation of CMPR, correct for the game's own textures and about 4% off
// for a PC-authored DXT5 file, which is what a replacement always is.

#include "../lib/gfx/texture_convert.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

constexpr uint16_t kRed565 = 0xF800;   // (255, 0, 0) once expanded
constexpr uint16_t kBlue565 = 0x001F;  // (0, 0, 255)

// One 16-byte BC3 block: an 8-byte BC4 alpha block then an 8-byte BC1 colour
// block. `alphaIdx` and `colourIdx` are the packed index words, built by
// repeat3/repeat2 below.
std::vector<uint8_t> Bc3Block(uint8_t alpha0, uint8_t alpha1, uint64_t alphaIdx, uint16_t colour0,
                              uint16_t colour1, uint32_t colourIdx) {
  std::vector<uint8_t> block(16);
  block[0] = alpha0;
  block[1] = alpha1;
  for (int i = 0; i < 6; ++i) {
    block[2 + i] = static_cast<uint8_t>((alphaIdx >> (8 * i)) & 0xFF);
  }
  block[8] = static_cast<uint8_t>(colour0 & 0xFF);
  block[9] = static_cast<uint8_t>(colour0 >> 8);
  block[10] = static_cast<uint8_t>(colour1 & 0xFF);
  block[11] = static_cast<uint8_t>(colour1 >> 8);
  for (int i = 0; i < 4; ++i) {
    block[12 + i] = static_cast<uint8_t>((colourIdx >> (8 * i)) & 0xFF);
  }
  return block;
}

/// The same 3-bit alpha index in all 16 texels.
uint64_t Repeat3(uint64_t index) {
  uint64_t out = 0;
  for (int i = 0; i < 16; ++i) {
    out |= index << (3 * i);
  }
  return out;
}

/// The same 2-bit colour index in all 16 texels.
uint32_t Repeat2(uint32_t index) {
  uint32_t out = 0;
  for (int i = 0; i < 16; ++i) {
    out |= index << (2 * i);
  }
  return out;
}

struct Rgba {
  uint8_t r, g, b, a;
  bool operator==(const Rgba&) const = default;
};

/// Decode a single 4x4 block and return every texel, so a decoder that only
/// gets the first one right cannot pass.
std::vector<Rgba> DecodeBc3Block(const std::vector<uint8_t>& block) {
  const auto out = aurora::gfx::decompress_bc_to_rgba8(wgpu::TextureFormat::BC3RGBAUnorm, 4, 4, 1,
                                                       {block.data(), block.size()});
  std::vector<Rgba> texels;
  if (out.size() != 4 * 4 * 4) {
    return texels;
  }
  for (int i = 0; i < 16; ++i) {
    const uint8_t* p = out.data() + i * 4;
    texels.push_back(Rgba{p[0], p[1], p[2], p[3]});
  }
  return texels;
}

void ExpectAllTexels(const std::vector<Rgba>& texels, Rgba want) {
  ASSERT_EQ(texels.size(), 16u);
  for (size_t i = 0; i < texels.size(); ++i) {
    EXPECT_EQ(texels[i], want) << "texel " << i;
  }
}

// alpha0 > alpha1 selects the eight-value interpolation, so index 0 is opaque.
TEST(Bc3Decode, ColourIndexZeroIsTheFirstColour) {
  ExpectAllTexels(DecodeBc3Block(Bc3Block(255, 0, 0, kRed565, kBlue565, Repeat2(0))),
                  Rgba{255, 0, 0, 255});
}

TEST(Bc3Decode, ColourIndexOneIsTheSecondColour) {
  ExpectAllTexels(DecodeBc3Block(Bc3Block(255, 0, 0, kRed565, kBlue565, Repeat2(1))),
                  Rgba{0, 0, 255, 255});
}

// The interpolated colours are the S3TC spec's thirds — (2*c0 + c1)/3 — and
// NOT aurora's GameCube S3TCBlend(), which would give 159/95 here.
TEST(Bc3Decode, InterpolatedColourUsesS3tcThirdsNotTheGameCubeBlend) {
  ExpectAllTexels(DecodeBc3Block(Bc3Block(255, 0, 0, kRed565, kBlue565, Repeat2(2))),
                  Rgba{170, 0, 85, 255});
  ExpectAllTexels(DecodeBc3Block(Bc3Block(255, 0, 0, kRed565, kBlue565, Repeat2(3))),
                  Rgba{85, 0, 170, 255});
}

TEST(Bc3Decode, AlphaIndexOneIsTheSecondAlpha) {
  ExpectAllTexels(DecodeBc3Block(Bc3Block(255, 0, Repeat3(1), kRed565, kBlue565, Repeat2(0))),
                  Rgba{255, 0, 0, 0});
}

// Eight-value mode: index 2 = (6*a0 + 1*a1)/7 = 1530/7 = 218.
TEST(Bc3Decode, EightValueAlphaInterpolation) {
  ExpectAllTexels(DecodeBc3Block(Bc3Block(255, 0, Repeat3(2), kRed565, kBlue565, Repeat2(0))),
                  Rgba{255, 0, 0, 218});
}

// alpha0 <= alpha1 selects the six-value mode, where index 6 is fully
// transparent and index 7 fully opaque regardless of the endpoints.
TEST(Bc3Decode, SixValueAlphaModeHasFixedEndpoints) {
  ExpectAllTexels(DecodeBc3Block(Bc3Block(0, 255, Repeat3(6), kRed565, kBlue565, Repeat2(0))),
                  Rgba{255, 0, 0, 0});
  ExpectAllTexels(DecodeBc3Block(Bc3Block(0, 255, Repeat3(7), kRed565, kBlue565, Repeat2(0))),
                  Rgba{255, 0, 0, 255});
}

TEST(Bc1Decode, DecodesAFullBlock) {
  // BC1 is 8 bytes: two endpoints and a 32-bit index word.
  std::vector<uint8_t> block(8);
  block[0] = static_cast<uint8_t>(kRed565 & 0xFF);
  block[1] = static_cast<uint8_t>(kRed565 >> 8);
  block[2] = static_cast<uint8_t>(kBlue565 & 0xFF);
  block[3] = static_cast<uint8_t>(kBlue565 >> 8);
  const auto out = aurora::gfx::decompress_bc_to_rgba8(wgpu::TextureFormat::BC1RGBAUnorm, 4, 4, 1,
                                                       {block.data(), block.size()});
  ASSERT_EQ(out.size(), 4u * 4u * 4u);
  EXPECT_EQ(out.data()[0], 255);  // index 0 -> first endpoint
  EXPECT_EQ(out.data()[2], 0);
}

// A truncated file must be refused, not read off the end of the buffer.
TEST(BcDecode, RejectsInputTooSmallForTheDeclaredSize) {
  const std::vector<uint8_t> half(8);
  EXPECT_TRUE(aurora::gfx::decompress_bc_to_rgba8(wgpu::TextureFormat::BC3RGBAUnorm, 4, 4, 1,
                                                  {half.data(), half.size()})
                  .empty());
  const std::vector<uint8_t> tiny(4);
  EXPECT_TRUE(aurora::gfx::decompress_bc_to_rgba8(wgpu::TextureFormat::BC1RGBAUnorm, 4, 4, 1,
                                                  {tiny.data(), tiny.size()})
                  .empty());
}

// Formats with no decoder return empty so the caller rejects the file rather
// than uploading garbage. BC2 is deliberately here: it shares BC3's block size,
// so decoding it as BC3 would produce plausible-looking wrong alpha.
TEST(BcDecode, ReturnsEmptyForFormatsWithNoDecoder) {
  const std::vector<uint8_t> block(64);
  for (const auto format : {wgpu::TextureFormat::BC7RGBAUnorm, wgpu::TextureFormat::BC5RGUnorm,
                            wgpu::TextureFormat::BC2RGBAUnorm, wgpu::TextureFormat::RGBA8Unorm}) {
    EXPECT_TRUE(
        aurora::gfx::decompress_bc_to_rgba8(format, 4, 4, 1, {block.data(), block.size()}).empty())
        << "format " << static_cast<uint32_t>(format);
  }
}

// Mip chains are packed one level after another; the decoder must walk all of
// them, not just the base.
TEST(Bc3Decode, DecodesAMipChain) {
  // 8x8 + 4x4 + 2x2 + 1x1 -> 4 + 1 + 1 + 1 blocks of 16 bytes.
  std::vector<uint8_t> data;
  for (int i = 0; i < 7; ++i) {
    const auto block = Bc3Block(255, 0, 0, kRed565, kBlue565, Repeat2(0));
    data.insert(data.end(), block.begin(), block.end());
  }
  const auto out = aurora::gfx::decompress_bc_to_rgba8(wgpu::TextureFormat::BC3RGBAUnorm, 8, 8, 4,
                                                       {data.data(), data.size()});
  // 64 + 16 + 4 + 1 texels.
  EXPECT_EQ(out.size(), (64u + 16u + 4u + 1u) * 4u);
}

}  // namespace
