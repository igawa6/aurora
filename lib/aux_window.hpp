#pragma once

#ifdef AURORA_ENABLE_GX

#include <webgpu/webgpu_cpp.h>

union SDL_Event;

namespace aurora::auxwin {

/// Returns true if the event targets the auxiliary window and was consumed.
/// Called from the main event loop before regular processing.
bool filter_event(const SDL_Event& event);

/// Present the auxiliary surface. Render worker thread only, after queue
/// submit.
void present();

/// Like present() but uses a caller-owned encoder and queue instead of
/// creating per-frame GPU command objects (Android memory-pressure path).
void present_with_encoder(wgpu::CommandEncoder* enc, wgpu::Queue* submitQueue);

/// Like present() but records the blit into a caller-owned encoder and
/// submits via a caller-owned queue — no separate GPU-command-object allocs.
void present_with_encoder(wgpu::CommandEncoder* encoder, wgpu::Queue* submitQueue);

} // namespace aurora::auxwin

#endif
