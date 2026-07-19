#pragma once

#ifdef AURORA_ENABLE_GX

union SDL_Event;

namespace aurora::auxwin {

/// Returns true if the event targets the auxiliary window and was consumed.
/// Called from the main event loop before regular processing.
bool filter_event(const SDL_Event& event);

/// Present the auxiliary surface. Render worker thread only, after queue
/// submit.
void present();

} // namespace aurora::auxwin

#endif
