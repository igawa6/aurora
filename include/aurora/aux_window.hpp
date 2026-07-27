#pragma once

#include <cstdint>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace aurora::auxwin {

struct CreateInfo {
  const char* title = "Aurora Aux Window";
  uint32_t width = 608;
  uint32_t height = 448;
  int32_t posX = -1;         // -1 = window-manager placement
  int32_t posY = -1;
  int32_t displayIndex = -1; // >= 0: center on that display (overrides posX/posY)
};

/// Create the auxiliary display-only window. Requires the graphics device to be
/// initialized (i.e. after aurora_initialize). Main/game thread only.
bool create(const CreateInfo& info);

/// Tear down the auxiliary window and its surface. Main/game thread only.
void destroy();

bool is_open() noexcept;

/// Set the texture displayed on the auxiliary window for the current frame.
/// The view must remain valid through the frame's submission (per-frame pooled
/// snapshots from gfx::resolve_pass qualify). Call between begin/end frame.
void set_source(wgpu::TextureView view, uint32_t width, uint32_t height);

/// Darken the auxiliary window at PRESENT time: 0 = untouched, 1 = black.
/// Applied to the blit, not to the source picture, so the fade keeps running
/// on frames where set_source is not called (e.g. the companion freezes on its
/// last frame during a stage load). Callable from any thread.
void set_dim(float amount);

/// Ask for the next presented aux frame to be copied back to CPU memory.
/// One-shot: a normal frame costs nothing. Callable from any thread.
void request_capture();

/// Move the most recent completed capture into `out` (tightly packed RGBA8,
/// top row first) and clear it. False while none is ready. Callable from any
/// thread.
bool take_capture(std::vector<uint8_t>& out, uint32_t* width, uint32_t* height);

/// True once after the user requested closing the auxiliary window.
bool consume_close_request();

/// Attach an externally-managed native window (e.g. an Android Presentation
/// surface as an ANativeWindow*) as the auxiliary output instead of an SDL
/// window. Pass nullptr to detach. Ownership of the window reference transfers
/// to aurora (released with ANativeWindow_release on Android). Callable from
/// any thread; the surface is (re)created on the render worker at the next
/// frame. No-op on platforms without native-window support.
void set_native_window(void* nativeWindow, uint32_t width, uint32_t height);

/// Current auxiliary surface size in pixels. Returns false when no surface is
/// active. Thread-safe.
bool get_surface_size(uint32_t* width, uint32_t* height);

} // namespace aurora::auxwin
