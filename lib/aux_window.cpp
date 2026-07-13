#include <aurora/aux_window.hpp>

#ifdef AURORA_ENABLE_GX

#include "aux_window.hpp"
#include "dawn/BackendBinding.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"
#include "window.hpp"

#include <aurora/gfx.hpp>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

#ifdef __ANDROID__
#include <android/native_window.h>
#endif

#include <algorithm>
#include <atomic>
#include <mutex>

namespace aurora::auxwin {
namespace {
Module Log("aurora::auxwin");

struct Source {
  wgpu::TextureView view;
  uint32_t width = 0;
  uint32_t height = 0;
};

// Lock order: g_surfaceMutex before g_mutex.
// g_surfaceMutex serializes GPU use of the surface (encode/present, which can
// block inside GetCurrentTexture/Present) against surface teardown. g_mutex
// guards shared state and is only ever held briefly, so the game and UI
// threads (set_source, get_surface_size, staging) never stall behind a
// blocking present — holding one mutex across present caused whole-game
// freezes on Android when the bottom display hiccuped.
std::mutex g_surfaceMutex;
std::mutex g_mutex;
bool g_active = false;
SDL_Window* g_window = nullptr;
SDL_WindowID g_windowId = 0;
wgpu::Surface g_surface;
wgpu::SurfaceConfiguration g_surfaceConfig;
wgpu::Sampler g_sampler;
Source g_source;

std::atomic<bool> g_needsConfigure{false};
std::atomic<uint32_t> g_pendingWidth{0};
std::atomic<uint32_t> g_pendingHeight{0};
std::atomic<bool> g_closeRequested{false};

// Externally-managed native window (Android Presentation surface). Attach and
// detach requests are staged here and consumed on the render worker.
void* g_pendingNativeWindow = nullptr;
bool g_nativeWindowDirty = false;
bool g_pendingDetach = false;
void* g_currentNativeWindow = nullptr;

void release_native_window(void* window) {
#ifdef __ANDROID__
  if (window != nullptr) {
    ANativeWindow_release(static_cast<ANativeWindow*>(window));
  }
#else
  (void)window;
#endif
}

void configure_surface_locked(uint32_t width, uint32_t height) {
  if (!g_surface || width == 0 || height == 0) {
    return;
  }
  g_surfaceConfig.width = width;
  g_surfaceConfig.height = height;
  g_surface.Configure(&g_surfaceConfig);
}

} // namespace

bool create(const CreateInfo& info) {
  if (!webgpu::g_device || !webgpu::g_instance) {
    Log.error("create: graphics device not initialized");
    return false;
  }
  std::scoped_lock locks{g_surfaceMutex, g_mutex};
  if (g_window != nullptr) {
    return true;
  }

  SDL_WindowFlags flags = SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_RESIZABLE;
  Sint32 posX = info.posX < 0 ? SDL_WINDOWPOS_UNDEFINED : info.posX;
  Sint32 posY = info.posY < 0 ? SDL_WINDOWPOS_UNDEFINED : info.posY;
  if (info.displayIndex >= 0) {
    int displayCount = 0;
    SDL_DisplayID* displays = SDL_GetDisplays(&displayCount);
    if (displays != nullptr) {
      if (info.displayIndex < displayCount) {
        posX = SDL_WINDOWPOS_CENTERED_DISPLAY(displays[info.displayIndex]);
        posY = SDL_WINDOWPOS_CENTERED_DISPLAY(displays[info.displayIndex]);
      }
      SDL_free(displays);
    }
  }

  const auto props = SDL_CreateProperties();
  SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, info.title);
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, posX);
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, posY);
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, static_cast<Sint32>(info.width));
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, static_cast<Sint32>(info.height));
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, flags);
  SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_EXTERNAL_GRAPHICS_CONTEXT_BOOLEAN, true);
  g_window = SDL_CreateWindowWithProperties(props);
  SDL_DestroyProperties(props);
  if (g_window == nullptr) {
    Log.error("Failed to create aux window: {}", SDL_GetError());
    return false;
  }
  g_windowId = SDL_GetWindowID(g_window);

  const auto chainedDescriptor = webgpu::utils::SetupWindowAndGetSurfaceDescriptor(g_window);
  if (!chainedDescriptor) {
    Log.error("Failed to create aux surface descriptor");
    SDL_DestroyWindow(g_window);
    g_window = nullptr;
    return false;
  }
  const wgpu::SurfaceDescriptor surfaceDescriptor{
      .nextInChain = chainedDescriptor.get(),
      .label = "Aux Surface",
  };
  g_surface = webgpu::g_instance.CreateSurface(&surfaceDescriptor);
  if (!g_surface) {
    Log.error("Failed to create aux surface");
    SDL_DestroyWindow(g_window);
    g_window = nullptr;
    return false;
  }

  // Reuse the format negotiated for the main surface so g_CopyPipeline is
  // compatible; Fifo is universally supported.
  g_surfaceConfig = wgpu::SurfaceConfiguration{
      .device = webgpu::g_device,
      .format = webgpu::g_graphicsConfig.surfaceConfiguration.format,
      .usage = wgpu::TextureUsage::RenderAttachment,
      .presentMode = wgpu::PresentMode::Fifo,
  };
  int pixelWidth = 0;
  int pixelHeight = 0;
  SDL_GetWindowSizeInPixels(g_window, &pixelWidth, &pixelHeight);
  configure_surface_locked(static_cast<uint32_t>(pixelWidth), static_cast<uint32_t>(pixelHeight));

  const wgpu::SamplerDescriptor samplerDescriptor{
      .label = "Aux Blit Sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  g_sampler = webgpu::g_device.CreateSampler(&samplerDescriptor);

  g_closeRequested = false;
  g_needsConfigure = false;
  g_active = true;
  Log.info("Aux window created ({}x{} px)", pixelWidth, pixelHeight);
  return true;
}

void destroy() {
  {
    std::lock_guard lock{g_mutex};
    if (g_window == nullptr) {
      return;
    }
    g_active = false;
    g_source = {};
  }
  // Drain the render worker so encode()/present() can no longer touch the
  // surface before it is torn down.
  gfx::synchronize();
  std::scoped_lock locks{g_surfaceMutex, g_mutex};
  if (g_surface) {
    g_surface.Unconfigure();
    g_surface = {};
  }
  g_sampler = {};
  if (g_window != nullptr) {
    SDL_DestroyWindow(g_window);
    g_window = nullptr;
    g_windowId = 0;
  }
  Log.info("Aux window destroyed");
}

bool is_open() noexcept {
  std::lock_guard lock{g_mutex};
  return g_active;
}

void set_source(wgpu::TextureView view, uint32_t width, uint32_t height) {
  std::lock_guard lock{g_mutex};
  g_source = Source{std::move(view), width, height};
}

bool consume_close_request() { return g_closeRequested.exchange(false); }

bool get_surface_size(uint32_t* width, uint32_t* height) {
  std::lock_guard lock{g_mutex};
  if (!g_active || !g_surface) {
    return false;
  }
  *width = g_surfaceConfig.width;
  *height = g_surfaceConfig.height;
  return true;
}

void set_native_window(void* nativeWindow, uint32_t width, uint32_t height) {
  if (nativeWindow == nullptr) {
    // Stage the detach — do NOT block on g_surfaceMutex. Synchronous
    // detach would stall the Android UI thread while the render worker
    // holds the surface mutex inside GetCurrentTexture/Present, and
    // accumulated stalls trigger ANR kills on Android.
    std::lock_guard lock{g_mutex};
    if (g_nativeWindowDirty && g_pendingNativeWindow != nullptr) {
      release_native_window(g_pendingNativeWindow);
      g_pendingNativeWindow = nullptr;
    }
    g_nativeWindowDirty = false;
    g_pendingDetach = true;
    return;
  }
  std::lock_guard lock{g_mutex};
  // Drop a staged-but-unconsumed window that is being replaced.
  if (g_nativeWindowDirty && g_pendingNativeWindow != nullptr && g_pendingNativeWindow != nativeWindow) {
    release_native_window(g_pendingNativeWindow);
  }
  g_pendingNativeWindow = nativeWindow;
  g_pendingWidth = width;
  g_pendingHeight = height;
  g_nativeWindowDirty = true;
}

namespace {

// Render worker: apply a staged native-window attach/detach. Caller holds
// BOTH g_surfaceMutex and g_mutex.
void consume_native_window_locked() {
  // Detach takes priority — runs inside the surface lock so the render
  // worker cannot interleave with GPU use of the surface.
  if (g_pendingDetach) {
    g_pendingDetach = false;
    if (g_currentNativeWindow != nullptr) {
      g_source = {};
      if (g_surface && g_window == nullptr) {
        g_surface.Unconfigure();
        g_surface = {};
      }
      release_native_window(g_currentNativeWindow);
      g_currentNativeWindow = nullptr;
      g_active = g_window != nullptr;
    }
  }

  if (!g_nativeWindowDirty) {
    return;
  }
  g_nativeWindowDirty = false;

  if (g_surface && g_window == nullptr) {
    g_surface.Unconfigure();
    g_surface = {};
  }
  if (g_currentNativeWindow != nullptr && g_currentNativeWindow != g_pendingNativeWindow) {
    release_native_window(g_currentNativeWindow);
  }
  g_currentNativeWindow = g_pendingNativeWindow;
  g_pendingNativeWindow = nullptr;
  if (g_currentNativeWindow == nullptr) {
    g_active = g_window != nullptr;
    return;
  }

#ifdef __ANDROID__
  wgpu::SurfaceSourceAndroidNativeWindow source;
  source.window = g_currentNativeWindow;
  const wgpu::SurfaceDescriptor surfaceDescriptor{
      .nextInChain = &source,
      .label = "Aux Surface (native window)",
  };
  g_surface = webgpu::g_instance.CreateSurface(&surfaceDescriptor);
  if (!g_surface) {
    Log.error("Failed to create aux surface from native window");
    release_native_window(g_currentNativeWindow);
    g_currentNativeWindow = nullptr;
    g_active = false;
    return;
  }
  g_surfaceConfig = wgpu::SurfaceConfiguration{
      .device = webgpu::g_device,
      .format = webgpu::g_graphicsConfig.surfaceConfiguration.format,
      .usage = wgpu::TextureUsage::RenderAttachment,
      .presentMode = wgpu::PresentMode::Fifo,
  };
  configure_surface_locked(g_pendingWidth.load(), g_pendingHeight.load());
  if (!g_sampler) {
    const wgpu::SamplerDescriptor samplerDescriptor{
        .label = "Aux Blit Sampler",
        .addressModeU = wgpu::AddressMode::ClampToEdge,
        .addressModeV = wgpu::AddressMode::ClampToEdge,
        .addressModeW = wgpu::AddressMode::ClampToEdge,
        .magFilter = wgpu::FilterMode::Linear,
        .minFilter = wgpu::FilterMode::Linear,
    };
    g_sampler = webgpu::g_device.CreateSampler(&samplerDescriptor);
  }
  g_active = true;
  Log.info("Aux native window attached ({}x{})", g_pendingWidth.load(), g_pendingHeight.load());
#else
  Log.warn("set_native_window: unsupported on this platform");
  release_native_window(g_currentNativeWindow);
  g_currentNativeWindow = nullptr;
#endif
}

} // namespace

bool filter_event(const SDL_Event& event) {
  if (event.type < SDL_EVENT_WINDOW_FIRST || event.type > SDL_EVENT_WINDOW_LAST) {
    return false;
  }
  {
    std::lock_guard lock{g_mutex};
    if (g_windowId == 0 || event.window.windowID != g_windowId) {
      return false;
    }
  }
  switch (event.type) {
  case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    g_pendingWidth = static_cast<uint32_t>(event.window.data1);
    g_pendingHeight = static_cast<uint32_t>(event.window.data2);
    g_needsConfigure = true;
    break;
  case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
    g_closeRequested = true;
    break;
  default:
    break;
  }
  return true;
}

void present() {
  // Single critical section for ALL GPU use of the aux surface — acquire,
  // blit (own encoder + submit), and present. Previously the blit was
  // recorded into the frame encoder in one lock scope and submitted outside
  // it, so a synchronous native-window detach (Android screen-off) could
  // destroy the surface between the two and crash the submit.
  std::lock_guard surfaceLock{g_surfaceMutex};
  wgpu::Surface surface;
  wgpu::Sampler sampler;
  Source source;
  uint32_t surfaceWidth = 0;
  uint32_t surfaceHeight = 0;
  {
    std::lock_guard lock{g_mutex};
    consume_native_window_locked();
    if (!g_active || !g_surface) {
      return;
    }
    if (g_needsConfigure.exchange(false)) {
      configure_surface_locked(g_pendingWidth.load(), g_pendingHeight.load());
    }
    surface = g_surface;
    sampler = g_sampler;
    source = g_source;
    surfaceWidth = g_surfaceConfig.width;
    surfaceHeight = g_surfaceConfig.height;
  }

  wgpu::SurfaceTexture surfaceTexture;
  surface.GetCurrentTexture(&surfaceTexture);
  switch (surfaceTexture.status) {
  case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
    break;
  case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
  case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
    g_needsConfigure = true;
    return;
  default:
    Log.warn("Aux surface texture unavailable: {}", static_cast<int>(surfaceTexture.status));
    return;
  }
  const auto view = surfaceTexture.texture.CreateView();

  const wgpu::CommandEncoderDescriptor encoderDescriptor{.label = "Aux window encoder"};
  const auto encoder = webgpu::g_device.CreateCommandEncoder(&encoderDescriptor);
  {
    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = view,
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = {0.0, 0.0, 0.0, 1.0},
        },
    };
    const wgpu::RenderPassDescriptor renderPassDescriptor{
        .label = "Aux window blit pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
    if (source.view && source.width != 0 && source.height != 0 && surfaceWidth != 0 &&
        surfaceHeight != 0)
    {
      // Aspect-fit letterbox; never stretch.
      const float scale = std::min(static_cast<float>(surfaceWidth) / static_cast<float>(source.width),
                                   static_cast<float>(surfaceHeight) / static_cast<float>(source.height));
      const float viewWidth = static_cast<float>(source.width) * scale;
      const float viewHeight = static_cast<float>(source.height) * scale;
      const float viewLeft = (static_cast<float>(surfaceWidth) - viewWidth) * 0.5f;
      const float viewTop = (static_cast<float>(surfaceHeight) - viewHeight) * 0.5f;
      const webgpu::TextureWithSampler sourceBinding{
          .view = source.view,
          .sampler = sampler,
      };
      pass.SetPipeline(webgpu::g_CopyPipeline);
      pass.SetBindGroup(0, webgpu::create_copy_bind_group(sourceBinding), 0, nullptr);
      pass.SetViewport(viewLeft, viewTop, viewWidth, viewHeight, 0.f, 1.f);
      pass.Draw(3);
    }
    pass.End();
  }
  const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Aux window command buffer"};
  const auto buffer = encoder.Finish(&cmdBufDescriptor);
  webgpu::g_queue.Submit(1, &buffer);

  const wgpu::ConvertibleStatus status = surface.Present();
  if (!status) {
    Log.warn("Aux surface present failed");
    g_needsConfigure = true;
  }
}
void present_with_encoder(wgpu::CommandEncoder* enc, wgpu::Queue* submitQueue) {
  // Single critical section for ALL GPU use of the aux surface — acquire,
  // blit (own encoder + submit), and present. Previously the blit was
  // recorded into the frame encoder in one lock scope and submitted outside
  // it, so a synchronous native-window detach (Android screen-off) could
  // destroy the surface between the two and crash the submit.
  std::lock_guard surfaceLock{g_surfaceMutex};
  wgpu::Surface surface;
  wgpu::Sampler sampler;
  Source source;
  uint32_t surfaceWidth = 0;
  uint32_t surfaceHeight = 0;
  {
    std::lock_guard lock{g_mutex};
    consume_native_window_locked();
    if (!g_active || !g_surface) {
      return;
    }
    if (g_needsConfigure.exchange(false)) {
      configure_surface_locked(g_pendingWidth.load(), g_pendingHeight.load());
    }
    surface = g_surface;
    sampler = g_sampler;
    source = g_source;
    surfaceWidth = g_surfaceConfig.width;
    surfaceHeight = g_surfaceConfig.height;
  }

  wgpu::SurfaceTexture surfaceTexture;
  surface.GetCurrentTexture(&surfaceTexture);
  switch (surfaceTexture.status) {
  case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
    break;
  case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
  case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
    g_needsConfigure = true;
    return;
  default:
    Log.warn("Aux surface texture unavailable: {}", static_cast<int>(surfaceTexture.status));
    return;
  }
  const auto view = surfaceTexture.texture.CreateView();

  // Uses caller-owned encoder (no per-frame CommandEncoder alloc).
  {
    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = view,
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = {0.0, 0.0, 0.0, 1.0},
        },
    };
    const wgpu::RenderPassDescriptor renderPassDescriptor{
        .label = "Aux window blit pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = enc->BeginRenderPass(&renderPassDescriptor);
    if (source.view && source.width != 0 && source.height != 0 && surfaceWidth != 0 &&
        surfaceHeight != 0)
    {
      // Aspect-fit letterbox; never stretch.
      const float scale = std::min(static_cast<float>(surfaceWidth) / static_cast<float>(source.width),
                                   static_cast<float>(surfaceHeight) / static_cast<float>(source.height));
      const float viewWidth = static_cast<float>(source.width) * scale;
      const float viewHeight = static_cast<float>(source.height) * scale;
      const float viewLeft = (static_cast<float>(surfaceWidth) - viewWidth) * 0.5f;
      const float viewTop = (static_cast<float>(surfaceHeight) - viewHeight) * 0.5f;
      const webgpu::TextureWithSampler sourceBinding{
          .view = source.view,
          .sampler = sampler,
      };
      pass.SetPipeline(webgpu::g_CopyPipeline);
      pass.SetBindGroup(0, webgpu::create_copy_bind_group(sourceBinding), 0, nullptr);
      pass.SetViewport(viewLeft, viewTop, viewWidth, viewHeight, 0.f, 1.f);
      pass.Draw(3);
    }
    pass.End();
  }
  const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Aux window command buffer"};
  const auto buffer = enc->Finish(&cmdBufDescriptor);
  submitQueue->Submit(1, &buffer);

  const wgpu::ConvertibleStatus status = surface.Present();
  if (!status) {
    Log.warn("Aux surface present failed");
    g_needsConfigure = true;
  }
}

} // namespace aurora::auxwin

#endif
