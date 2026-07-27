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
#include <cstring>
#include <mutex>
#include <vector>

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

// Present-time dim, 0..1. Deliberately NOT part of Source: the caller stops
// refreshing the source texture while the companion's HUD is torn down during
// a stage load, and the fade has to keep running across those frames.
std::atomic<float> g_dim{0.0f};

// One-shot CPU read-back of the presented aux frame, for the on-device
// screenshot path. Idle until request_capture(); the blit is then repeated
// into g_captureTexture (the surface texture is RenderAttachment-only and
// cannot be a copy source) and copied to a staging buffer.
enum class CaptureState { Idle, Armed, InFlight, Ready };
std::mutex g_captureMutex;
CaptureState g_captureState = CaptureState::Idle;
wgpu::Texture g_captureTexture;
wgpu::TextureView g_captureView;
wgpu::Buffer g_captureStaging;
uint32_t g_captureWidth = 0;
uint32_t g_captureHeight = 0;
uint32_t g_capturePaddedBpr = 0;
wgpu::TextureFormat g_captureFormat = wgpu::TextureFormat::Undefined;
bool g_captureSwizzleBgr = false;
std::vector<uint8_t> g_captureData;
// Bumped whenever a capture is abandoned (teardown, detach). The map callback
// carries the generation it was issued under and drops itself if they differ,
// so an in-flight read-back can never resurrect state that has been released
// and can never wedge the request path.
uint64_t g_captureGeneration = 0;

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

// Render worker: (re)allocate the capture target and staging buffer for the
// current surface size. Caller holds g_captureMutex.
//
// NOTE: the null tests below do NOT detect an allocation failure — Dawn returns
// a valid *error object*, not a null handle, so a genuine OOM surfaces later as
// an uncaptured device error. They only guard a device that is already gone.
// This is a debug-only capture path, so that is accepted rather than wrapped in
// an error scope.
bool ensure_capture_locked(uint32_t width, uint32_t height, wgpu::TextureFormat format) {
  constexpr uint32_t kCopyAlign = 256;  // WebGPU bytesPerRow alignment
  const uint32_t paddedBpr = (width * 4 + kCopyAlign - 1) / kCopyAlign * kCopyAlign;
  if (g_captureTexture && g_captureWidth == width && g_captureHeight == height &&
      g_captureFormat == format) {
    return true;
  }
  // Partial reuse is not worth the bookkeeping; start from nothing so a failed
  // allocation cannot leave half-valid dimensions behind.
  g_captureTexture = {};
  g_captureView = {};
  g_captureStaging = {};
  g_captureWidth = 0;
  g_captureHeight = 0;
  g_capturePaddedBpr = 0;
  const wgpu::TextureDescriptor textureDescriptor{
      .label = "Aux Capture Texture",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .dimension = wgpu::TextureDimension::e2D,
      .size = wgpu::Extent3D{width, height, 1},
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  g_captureTexture = webgpu::g_device.CreateTexture(&textureDescriptor);
  if (!g_captureTexture) {
    return false;
  }
  g_captureView = g_captureTexture.CreateView();
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Aux Capture Staging Buffer",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = static_cast<uint64_t>(paddedBpr) * height,
  };
  g_captureStaging = webgpu::g_device.CreateBuffer(&bufferDescriptor);
  if (!g_captureStaging) {
    g_captureTexture = {};
    g_captureView = {};
    return false;
  }
  g_captureFormat = format;
  g_captureWidth = width;
  g_captureHeight = height;
  g_capturePaddedBpr = paddedBpr;
  // Surfaces are commonly BGRA8; the public capture is always RGBA8.
  g_captureSwizzleBgr = format == wgpu::TextureFormat::BGRA8Unorm ||
                        format == wgpu::TextureFormat::BGRA8UnormSrgb;
  return true;
}

// Abandon any capture and drop its resources. Caller holds g_captureMutex.
//
// Bumping the generation is what makes this safe: the pending map callback owns
// its own reference to the staging buffer, so clearing the global here cannot
// destroy the buffer and therefore cannot make Dawn fire an aborted callback
// INLINE on this thread — which, on a non-recursive mutex we already hold,
// deadlocked the game thread.
void release_capture_locked() {
  g_captureGeneration++;
  g_captureState = CaptureState::Idle;
  g_captureTexture = {};
  g_captureView = {};
  g_captureStaging = {};
  g_captureWidth = 0;
  g_captureHeight = 0;
  g_capturePaddedBpr = 0;
  g_captureData.clear();
  g_captureData.shrink_to_fit();
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
  {
    // gfx::synchronize() above drains the render worker, but does NOT wait on
    // the GPU — a map may still be pending here, which is exactly what
    // release_capture_locked is built to survive.
    std::lock_guard captureLock{g_captureMutex};
    release_capture_locked();
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

void set_dim(float amount) {
  g_dim.store(amount < 0.0f ? 0.0f : (amount > 1.0f ? 1.0f : amount));
}

void request_capture() {
  std::lock_guard lock{g_captureMutex};
  // Always re-armable. A copy already in flight is abandoned rather than
  // waited on (the generation bump makes its completion a no-op), so a capture
  // that never completes — surface torn down mid-map, display detached — can
  // no longer wedge this path permanently.
  if (g_captureState == CaptureState::InFlight) {
    g_captureGeneration++;
  }
  g_captureData.clear();
  g_captureState = CaptureState::Armed;
}

bool take_capture(std::vector<uint8_t>& out, uint32_t* width, uint32_t* height) {
  std::lock_guard lock{g_captureMutex};
  if (g_captureState != CaptureState::Ready) {
    return false;
  }
  out = std::move(g_captureData);
  g_captureData.clear();
  if (width != nullptr) {
    *width = g_captureWidth;
  }
  if (height != nullptr) {
    *height = g_captureHeight;
  }
  g_captureState = CaptureState::Idle;
  return true;
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
      {
        // Android never calls destroy(), so this is the ONLY cleanup path
        // there: without it a single screenshot left ~16 MB of texture plus
        // host-visible staging buffer resident for the rest of the process,
        // across every subsequent detach/reattach.
        std::lock_guard captureLock{g_captureMutex};
        release_capture_locked();
      }
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
  wgpu::TextureFormat surfaceFormat = wgpu::TextureFormat::Undefined;
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
    surfaceFormat = g_surfaceConfig.format;
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
  const bool haveSource = source.view && source.width != 0 && source.height != 0 &&
                          surfaceWidth != 0 && surfaceHeight != 0;
  // Aspect-fit letterbox; never stretch. Shared by the surface blit and the
  // capture blit so a screenshot matches the panel pixel for pixel.
  float viewLeft = 0.f, viewTop = 0.f, viewWidth = 0.f, viewHeight = 0.f;
  if (haveSource) {
    const float scale = std::min(static_cast<float>(surfaceWidth) / static_cast<float>(source.width),
                                 static_cast<float>(surfaceHeight) / static_cast<float>(source.height));
    viewWidth = static_cast<float>(source.width) * scale;
    viewHeight = static_cast<float>(source.height) * scale;
    viewLeft = (static_cast<float>(surfaceWidth) - viewWidth) * 0.5f;
    viewTop = (static_cast<float>(surfaceHeight) - viewHeight) * 0.5f;
  }
  const webgpu::TextureWithSampler sourceBinding{
      .view = source.view,
      .sampler = sampler,
  };
  // out = src * constant. The letterbox stays at the pass clear (black), so a
  // full dim reaches true black everywhere.
  //
  // NOTE: this multiplies the surface's stored values. best_surface_format()
  // prefers non-sRGB RGBA8/BGRA8Unorm, so the fade lands on sRGB-ENCODED values
  // and reads as perceptually linear. On a fallback path that selected an
  // *UnormSrgb format, WebGPU would blend in linear space and the identical
  // constant would look like a noticeably slower fade.
  const float dim = g_dim.load();
  const double keep = 1.0 - static_cast<double>(dim);
  const wgpu::Color blendConstant{keep, keep, keep, 1.0};
  const bool dimmed = dim > 0.001f && webgpu::g_CopyDimPipeline != nullptr;

  // Built once: a capture frame records the same blit into two passes, and
  // allocating a second identical bind group for the copy was pure waste.
  const wgpu::BindGroup blitBindGroup =
      haveSource ? webgpu::create_copy_bind_group(sourceBinding) : wgpu::BindGroup{};
  const auto record_blit = [&](const wgpu::RenderPassEncoder& pass) {
    if (!haveSource) {
      return;
    }
    pass.SetPipeline(dimmed ? webgpu::g_CopyDimPipeline : webgpu::g_CopyPipeline);
    if (dimmed) {
      pass.SetBlendConstant(&blendConstant);
    }
    pass.SetBindGroup(0, blitBindGroup, 0, nullptr);
    pass.SetViewport(viewLeft, viewTop, viewWidth, viewHeight, 0.f, 1.f);
    pass.Draw(3);
  };

  {
    const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
    record_blit(pass);
    pass.End();
  }

  // Screenshot: repeat the same blit into an owned CopySrc texture (the
  // surface texture is RenderAttachment-only) and stage it for read-back.
  size_t captureMapSize = 0;
  wgpu::Buffer captureStaging;
  uint64_t captureGen = 0;
  uint32_t captureW = 0, captureH = 0, captureBpr = 0;
  bool captureSwizzle = false;
  {
    std::lock_guard captureLock{g_captureMutex};
    if (g_captureState == CaptureState::Armed && haveSource &&
        ensure_capture_locked(surfaceWidth, surfaceHeight, surfaceFormat))
    {
      const std::array captureAttachments{
          wgpu::RenderPassColorAttachment{
              .view = g_captureView,
              .loadOp = wgpu::LoadOp::Clear,
              .storeOp = wgpu::StoreOp::Store,
              .clearValue = {0.0, 0.0, 0.0, 1.0},
          },
      };
      const wgpu::RenderPassDescriptor capturePassDescriptor{
          .label = "Aux window capture pass",
          .colorAttachmentCount = captureAttachments.size(),
          .colorAttachments = captureAttachments.data(),
      };
      const auto capturePass = encoder.BeginRenderPass(&capturePassDescriptor);
      record_blit(capturePass);
      capturePass.End();

      const wgpu::TexelCopyTextureInfo copySrc{.texture = g_captureTexture};
      const wgpu::TexelCopyBufferInfo copyDst{
          .layout =
              wgpu::TexelCopyBufferLayout{
                  .bytesPerRow = g_capturePaddedBpr,
                  .rowsPerImage = g_captureHeight,
              },
          .buffer = g_captureStaging,
      };
      const wgpu::Extent3D copySize{g_captureWidth, g_captureHeight, 1};
      encoder.CopyTextureToBuffer(&copySrc, &copyDst, &copySize);
      g_captureState = CaptureState::InFlight;
      captureMapSize = static_cast<size_t>(g_capturePaddedBpr) * g_captureHeight;
      captureStaging = g_captureStaging;
      captureGen = g_captureGeneration;
      captureW = g_captureWidth;
      captureH = g_captureHeight;
      captureBpr = g_capturePaddedBpr;
      captureSwizzle = g_captureSwizzleBgr;
    }
  }

  const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Aux window command buffer"};
  const auto buffer = encoder.Finish(&cmdBufDescriptor);
  webgpu::g_queue.Submit(1, &buffer);

  if (captureMapSize != 0) {
    // Called OUTSIDE g_captureMutex: the callback takes it, and MapAsync's
    // error path completes inline under AllowSpontaneous.
    //
    // The lambda captures the staging buffer BY VALUE, so it holds its own
    // reference for as long as the map is outstanding. That is what lets
    // release_capture_locked() drop the global without destroying the buffer
    // and triggering an inline aborted callback into a mutex we already hold.
    // It also snapshots the dimensions, so a resize mid-map cannot make the
    // unpack read with the wrong stride.
    // AllowSpontaneous: gfx's per-frame g_instance.ProcessEvents() drives it.
    captureStaging.MapAsync(
        wgpu::MapMode::Read, 0, captureMapSize, wgpu::CallbackMode::AllowSpontaneous,
        [captureStaging, captureGen, captureW, captureH, captureBpr, captureSwizzle](
            wgpu::MapAsyncStatus status, wgpu::StringView message) {
          std::lock_guard lock{g_captureMutex};
          if (captureGen != g_captureGeneration) {
            // Superseded or released while in flight; nothing to publish.
            if (status == wgpu::MapAsyncStatus::Success) {
              captureStaging.Unmap();
            }
            return;
          }
          if (status != wgpu::MapAsyncStatus::Success) {
            if (status != wgpu::MapAsyncStatus::CallbackCancelled &&
                status != wgpu::MapAsyncStatus::Aborted) {
              Log.warn("Aux capture read-back failed: {}", message);
            }
            g_captureState = CaptureState::Idle;
            return;
          }
          const size_t mappedSize = static_cast<size_t>(captureBpr) * captureH;
          const auto* src =
              static_cast<const uint8_t*>(captureStaging.GetConstMappedRange(0, mappedSize));
          if (src != nullptr) {
            const size_t rowBytes = static_cast<size_t>(captureW) * 4;
            g_captureData.resize(rowBytes * captureH);
            for (uint32_t y = 0; y < captureH; ++y) {
              const uint8_t* in = src + static_cast<size_t>(y) * captureBpr;
              uint8_t* out = g_captureData.data() + static_cast<size_t>(y) * rowBytes;
              if (captureSwizzle) {
                for (uint32_t x = 0; x < captureW; ++x) {
                  out[x * 4 + 0] = in[x * 4 + 2];
                  out[x * 4 + 1] = in[x * 4 + 1];
                  out[x * 4 + 2] = in[x * 4 + 0];
                  out[x * 4 + 3] = 0xFF;  // the aux surface is opaque
                }
              } else {
                std::memcpy(out, in, rowBytes);
                for (uint32_t x = 0; x < captureW; ++x) {
                  out[x * 4 + 3] = 0xFF;
                }
              }
            }
            g_captureWidth = captureW;
            g_captureHeight = captureH;
            g_captureState = CaptureState::Ready;
          } else {
            g_captureState = CaptureState::Idle;
          }
          captureStaging.Unmap();
        });
  }

  const wgpu::ConvertibleStatus status = surface.Present();
  if (!status) {
    Log.warn("Aux surface present failed");
    g_needsConfigure = true;
  }
}

} // namespace aurora::auxwin

#endif
