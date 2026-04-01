/*
 * sealMega IDE - Hardware Constrained IPC
 * Module 1: Kernel-Bypass HugePages & Zero-Copy Named Pipes
 * 
 * Built this because I am hardware constrained and wanted to help my friends out.
 *
 * This C++ extension implements:
 * 1. SEC_LARGE_PAGES memory mapping. If you don't run as Admin with SE_LOCK_MEMORY_NAME, it crashes.
 * 2. An ultra-fast Windows Named Pipe (\\.\pipe\sealmega_tokens).
 *
 * God I hate Windows Kernel programming so much.
 */

#include <pybind11/pybind11.h>
#include <windows.h>
#include <iostream>
#include <string>

namespace py = pybind11;

// 10MB block. Large Pages require 2MB alignment.
// Pray to whatever deity you believe in that your RAM isn't fragmented.
constexpr SIZE_T IPC_SIZE = 10 * 1024 * 1024; // 10MB
void* g_hugepage_block = nullptr;
HANDLE g_token_pipe = INVALID_HANDLE_VALUE;

bool enable_large_pages_privilege() {
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
        std::cerr << "[HugePages] OpenProcessToken failed: " << GetLastError() << "\n";
        return false;
    }

    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
        std::cerr << "[HugePages] LookupPrivilegeValue failed: " << GetLastError() << "\n";
        CloseHandle(hToken);
        return false;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (!AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0)) {
        std::cerr << "[HugePages] AdjustTokenPrivileges failed: " << GetLastError() << "\n";
        CloseHandle(hToken);
        return false;
    }

    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED) {
        std::cerr << "[HugePages] The token does not have the specified privilege. Try running as Administrator.\n";
        CloseHandle(hToken);
        return false;
    }

    CloseHandle(hToken);
    return true;
}

bool init_hugepages_ipc() {
    if (g_hugepage_block != nullptr) return true; // Already initialized

    if (!enable_large_pages_privilege()) {
        std::cerr << "[HugePages] Failed to acquire SE_LOCK_MEMORY_NAME privilege. Falling back to standard VirtualAlloc.\n";
        // Fallback to standard memory if privileges missing, but complain loudly.
        g_hugepage_block = VirtualAlloc(NULL, IPC_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        return g_hugepage_block != nullptr;
    }

    SIZE_T min_page_size = GetLargePageMinimum();
    if (min_page_size == 0) {
        std::cerr << "[HugePages] Large pages are not supported by the processor. Falling back.\n";
        g_hugepage_block = VirtualAlloc(NULL, IPC_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        return g_hugepage_block != nullptr;
    }

    // Round up size to multiple of min_page_size
    SIZE_T alloc_size = (IPC_SIZE + min_page_size - 1) & ~(min_page_size - 1);

    std::cout << "[HugePages] Allocating " << alloc_size << " bytes using MEM_LARGE_PAGES (PageSize=" << min_page_size << ")\n";
    
    // Directive 1: SEC_LARGE_PAGES kernel bypass allocation
    g_hugepage_block = VirtualAlloc(NULL, alloc_size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);

    if (g_hugepage_block == nullptr) {
        std::cerr << "[HugePages] VirtualAlloc with MEM_LARGE_PAGES failed: " << GetLastError() << "\n";
        std::cerr << "[HugePages] NOTE: System RAM might be too fragmented. Reboot or fallback to standard pages.\n";
        // Final fallback
        g_hugepage_block = VirtualAlloc(NULL, IPC_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    } else {
        std::cout << "[HugePages] SUCCESS: Zero-TLB-Miss IPC block mapped at " << g_hugepage_block << "\n";
    }

    return g_hugepage_block != nullptr;
}

// Zero-copy token pump
// Because FastAPI adds 10ms of network overhead and I'd rather die than use it for streams.
bool create_token_pipe() {
    if (g_token_pipe != INVALID_HANDLE_VALUE) return true;

    std::wstring pipe_name = L"\\\\.\\pipe\\sealmega_tokens";
    std::wcout << L"[TokenPump] Creating 0-copy Named Pipe at " << pipe_name << L". Don't break the pipe.\n";

    g_token_pipe = CreateNamedPipeW(
        pipe_name.c_str(),
        PIPE_ACCESS_OUTBOUND, // 7B model only writes
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1, // 1 instance
        8192, // 8KB out buffer
        8192, // 8KB in buffer
        0,
        NULL // Security attrs
    );

    if (g_token_pipe == INVALID_HANDLE_VALUE) {
        std::cerr << "[TokenPump] Failed to create Named Pipe: " << GetLastError() << "\n";
        return false;
    }

    // Non-blocking connect listener? For simplicity, we assume client connects eventually.
    // In a real loop, you'd use Overlapped I/O here too.
    return true;
}

bool pump_token(const std::string& token) {
    if (g_token_pipe == INVALID_HANDLE_VALUE) {
        return false; // Pipe not alive
    }

    // If no one is connected, ConnectNamedPipe will block. 
    // We should poll or use PeekNamedPipe. Assuming connection is established for now.
    DWORD bytesWritten;
    BOOL success = WriteFile(
        g_token_pipe,
        token.c_str(),
        token.length(),
        &bytesWritten,
        NULL
    );

    if (!success) {
        // Disconnected
        DisconnectNamedPipe(g_token_pipe);
        return false;
    }

    return true;
}

PYBIND11_MODULE(extreme_hardware, m) {
    m.doc() = "Extreme Hardware Hacks (HugePages & 0-Copy Pipe)";
    m.def("init_hugepages_ipc", &init_hugepages_ipc, "Allocate 10MB IPC block using Kernel-Bypass MEM_LARGE_PAGES");
    m.def("create_token_pipe", &create_token_pipe, "Create the \\\\.\\pipe\\sealmega_tokens Named Pipe");
    m.def("pump_token", &pump_token, "Blast a single token string directly to VS Code memory space via Named Pipe");
}
