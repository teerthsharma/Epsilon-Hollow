/*
 * sealMega IDE - CUDA VRAM Semaphore Fencing
 * 
 * Built this because we are all hardware constrained and poor.
 * 
 * This C++ extension exposes fence_vram() and release_vram() to Python
 * via pybind11. When AirLLM begins a 70B layer swap, the 7B model's
 * sampling loop must poll this hardware indicator and yield.
 * 
 * On systems without CUDA, falls back to OS-level mutual exclusion
 * (still better than Python asyncio.Lock which does absolutely nothing to the GPU).
 * Yes, my Python lock crashed the GPU 17 times today. Here is the C++ version.
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <iostream>

// ─── Global Hardware Fence ───
// This atomic flag is the ONLY thing standing between
// a working IDE and CUDA_ERROR_OUT_OF_MEMORY. It's hanging by a thread.
static std::atomic<bool> g_vram_fenced{false};
static std::mutex g_fence_mutex;
static std::atomic<int> g_fence_holder{-1};  // -1 = nobody, 0 = foreman, 1 = logicgate, 2 = architect

// Tier IDs
constexpr int TIER_FOREMAN   = 0;
constexpr int TIER_LOGICGATE = 1;
constexpr int TIER_ARCHITECT = 2;

/**
 * Acquire the VRAM fence for a specific tier.
 * If another tier holds the fence, this call BLOCKS until released.
 * This is the C++ level lock - not a Python asyncio toy.
 */
bool fence_vram(int tier_id) {
    std::lock_guard<std::mutex> lock(g_fence_mutex);
    
    // If already fenced by another tier, we must wait
    while (g_vram_fenced.load() && g_fence_holder.load() != tier_id) {
        // Yield CPU to prevent busy-waiting from cooking the processor
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    g_vram_fenced.store(true);
    g_fence_holder.store(tier_id);
    return true;
}

/**
 * Release the VRAM fence. The tier that holds it must release it.
 */
bool release_vram(int tier_id) {
    if (g_fence_holder.load() != tier_id) {
        std::cerr << "[VRAMGuard] ERROR: Tier " << tier_id 
                  << " tried to release fence held by tier " 
                  << g_fence_holder.load() << std::endl;
        return false;
    }
    
    g_vram_fenced.store(false);
    g_fence_holder.store(-1);
    return true;
}

/**
 * Check if the VRAM is currently fenced (non-blocking).
 * The 7B model's token sampling loop must call this before every
 * forward pass. If true, it must SLEEP, not generate.
 */
bool is_vram_fenced() {
    return g_vram_fenced.load();
}

/**
 * Get which tier currently holds the fence.
 * Returns -1 if no tier holds it.
 */
int get_fence_holder() {
    return g_fence_holder.load();
}

// ─── CUDA-Specific Fencing (when CUDA is available) ───
#ifdef __CUDACC__
#include <cuda_runtime.h>

static cudaEvent_t g_fence_event;
static bool g_cuda_initialized = false;

bool cuda_fence_init() {
    if (!g_cuda_initialized) {
        cudaError_t err = cudaEventCreate(&g_fence_event);
        if (err != cudaSuccess) {
            std::cerr << "[VRAMGuard] CUDA event creation failed: " 
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        g_cuda_initialized = true;
    }
    return true;
}

bool cuda_fence_sync() {
    if (!g_cuda_initialized) return false;
    cudaEventRecord(g_fence_event);
    cudaEventSynchronize(g_fence_event);
    return true;
}

#else
// Fallback: no CUDA available
bool cuda_fence_init() {
    std::cerr << "[VRAMGuard] No CUDA detected. Using OS-level mutex only." << std::endl;
    return false;
}

bool cuda_fence_sync() {
    return false;
}
#endif


// ─── Python Binding via pybind11 ───
// Compile: pip install pybind11 && c++ -O2 -shared -std=c++17 -fPIC
//          $(python3 -m pybind11 --includes) vram_guard.cpp
//          -o vram_guard$(python3-config --extension-suffix)

#ifdef PYBIND11_AVAILABLE
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(vram_guard, m) {
    m.doc() = "sealMega VRAM Semaphore Fencing - Hardware Level GPU Lock";
    
    m.def("fence_vram", &fence_vram, 
          "Acquire VRAM fence for a tier (blocks if another tier holds it)",
          py::arg("tier_id"));
    m.def("release_vram", &release_vram,
          "Release VRAM fence",
          py::arg("tier_id"));
    m.def("is_vram_fenced", &is_vram_fenced,
          "Check if VRAM is currently fenced (non-blocking poll)");
    m.def("get_fence_holder", &get_fence_holder,
          "Get which tier currently holds the fence (-1 = none)");
    m.def("cuda_fence_init", &cuda_fence_init,
          "Initialize CUDA event for hardware-level sync");
    m.def("cuda_fence_sync", &cuda_fence_sync,
          "Synchronize on CUDA event (hard GPU barrier)");
    
    // Tier constants
    m.attr("TIER_FOREMAN") = TIER_FOREMAN;
    m.attr("TIER_LOGICGATE") = TIER_LOGICGATE;
    m.attr("TIER_ARCHITECT") = TIER_ARCHITECT;
}
#endif
