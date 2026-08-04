#pragma once

#include "texture.hpp"
#include <aurora/texture.hpp>
#include <optional>

namespace aurora::gfx::texture_replacement {
struct ReplacementResult {
  TextureHandle handle;
  uint64_t id = 0;
};

struct StreamingStats {
  uint64_t pendingLoads = 0;
  uint64_t publishes = 0;
  uint64_t publishBytes = 0;
};

void shutdown() noexcept;
/// Drop every cached replacement texture. Wired to SDL_EVENT_LOW_MEMORY: the
/// replacement cache is by far the largest discretionary allocation the process
/// holds, so it is the first thing to give back when the OS is under pressure.
/// Costs a reload of whatever is still on screen; the alternative is being
/// killed.
void on_low_memory() noexcept;
StreamingStats process_streaming() noexcept;
std::optional<ReplacementResult> find_pointer_replacement(const GXTexObj_& obj) noexcept;
std::optional<ReplacementResult> find_source_replacement(const GXTexObj_& obj,
                                                         const texture::TextureSourceKey& sourceKey) noexcept;
bool should_build_source_key() noexcept;
bool has_replacement(const GXTexObj_& obj) noexcept;
bool has_replacement(const GXTexObj_& obj, const GXTlutObj_& tlut) noexcept;
std::string build_texture_replacement_name(const GXTexObj_& obj) noexcept;
std::string build_texture_replacement_name(const texture::TextureSourceKey& sourceKey) noexcept;

namespace testing {
void set_workers_paused(bool paused) noexcept;
void set_worker_count(uint32_t count) noexcept;
bool wait_for_completions(uint64_t id, uint32_t count, uint32_t timeoutMs) noexcept;
} // namespace testing
} // namespace aurora::gfx::texture_replacement
