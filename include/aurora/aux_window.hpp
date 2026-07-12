#pragma once

#include <cstdint>

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

/// True once after the user requested closing the auxiliary window.
bool consume_close_request();

/// Attach an externally-managed native window (e.g. an Android Presentation
/// surface as an ANativeWindow*) as the auxiliary output instead of an SDL
/// window. Pass nullptr to detach. Ownership of the window reference transfers
/// to aurora (released with ANativeWindow_release on Android). Callable from
/// any thread; the surface is (re)created on the render worker at the next
/// frame. No-op on platforms without native-window support.
void set_native_window(void* nativeWindow, uint32_t width, uint32_t height);

} // namespace aurora::auxwin
