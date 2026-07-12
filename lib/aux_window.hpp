#pragma once

#ifdef AURORA_ENABLE_GX

#include <webgpu/webgpu_cpp.h>

union SDL_Event;

namespace aurora::auxwin {

/// Returns true if the event targets the auxiliary window and was consumed.
/// Called from the main event loop before regular processing.
bool filter_event(const SDL_Event& event);

/// Encode the blit of the current source texture onto the auxiliary surface.
/// Render worker thread only, inside the end-of-frame callback.
void encode(const wgpu::CommandEncoder& encoder);

/// Present the auxiliary surface. Render worker thread only, after queue
/// submit.
void present();

} // namespace aurora::auxwin

#endif
