#pragma once

#include "texture.hpp"

#include <aurora/texture.hpp>
#include <optional>

namespace aurora::gfx::texture_replacement {
void initialize() noexcept;
void shutdown() noexcept;
/// Drop every cached replacement texture. Wired to SDL_EVENT_LOW_MEMORY: the
/// replacement cache is by far the largest discretionary allocation the process
/// holds, so it is the first thing to give back when the OS is under pressure.
/// Costs a reload of whatever is still on screen; the alternative is being
/// killed.
void on_low_memory() noexcept;
/// Carries a texture's source key between frames.
///
/// Building that key means hashing the whole base level. That is fine once per
/// texture, but a texture whose replacement is still decoding gets looked up
/// again on EVERY frame until it lands — so without somewhere to keep the key,
/// a room's worth of pending textures re-hashes megabytes per frame for as long
/// as the decode takes. The owner is the caller's own per-texture cache entry,
/// which already knows when the texture data changed and can discard it.
struct SourceKeyCache {
  aurora::texture::TextureSourceKey key;
  bool valid = false;
};

/// Look up a replacement for `obj`.
///
/// `pending` (optional) is set when a background decode is in flight: the
/// replacement is not ready, the game's own texture should be drawn this frame,
/// and the caller must NOT cache that fallback against the texture object, or
/// the replacement will never get a chance to appear for it.
std::optional<TextureHandle> find_replacement(const GXTexObj_& obj, bool* pending = nullptr,
                                             SourceKeyCache* keyCache = nullptr) noexcept;
std::optional<TextureHandle> find_replacement(const GXTexObj_& obj, const GXTlutObj_& tlut,
                                             bool* pending = nullptr,
                                             SourceKeyCache* keyCache = nullptr) noexcept;
bool has_replacement(const GXTexObj_& obj) noexcept;
bool has_replacement(const GXTexObj_& obj, const GXTlutObj_& tlut) noexcept;
std::string build_texture_replacement_name(const GXTexObj_& obj) noexcept;
} // namespace aurora::gfx::texture_replacement
