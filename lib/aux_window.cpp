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

// Render-worker state between encode() and present()
wgpu::Texture g_acquiredTexture;

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
  std::lock_guard lock{g_mutex};
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
  std::lock_guard lock{g_mutex};
  g_acquiredTexture = {};
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

void encode(const wgpu::CommandEncoder& encoder) {
  wgpu::Surface surface;
  Source source;
  uint32_t surfaceWidth = 0;
  uint32_t surfaceHeight = 0;
  {
    std::lock_guard lock{g_mutex};
    if (!g_active || !g_surface) {
      return;
    }
    if (g_needsConfigure.exchange(false)) {
      configure_surface_locked(g_pendingWidth.load(), g_pendingHeight.load());
    }
    surface = g_surface;
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
  g_acquiredTexture = std::move(surfaceTexture.texture);
  const auto view = g_acquiredTexture.CreateView();

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
  if (source.view && source.width != 0 && source.height != 0) {
    const auto viewport =
        webgpu::calculate_present_viewport(surfaceWidth, surfaceHeight, source.width, source.height);
    const webgpu::TextureWithSampler sourceBinding{
        .view = source.view,
        .sampler = g_sampler,
    };
    pass.SetPipeline(webgpu::g_CopyPipeline);
    pass.SetBindGroup(0, webgpu::create_copy_bind_group(sourceBinding), 0, nullptr);
    pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
    pass.Draw(3);
  }
  pass.End();
}

void present() {
  wgpu::Surface surface;
  {
    std::lock_guard lock{g_mutex};
    if (!g_active || !g_surface || !g_acquiredTexture) {
      g_acquiredTexture = {};
      return;
    }
    surface = g_surface;
  }
  g_acquiredTexture = {};
  const wgpu::ConvertibleStatus status = surface.Present();
  if (!status) {
    Log.warn("Aux surface present failed");
    g_needsConfigure = true;
  }
}

} // namespace aurora::auxwin

#endif
