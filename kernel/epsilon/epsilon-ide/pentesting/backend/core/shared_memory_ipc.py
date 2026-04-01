"""
sealMega IDE — Zero-Copy IPC Manager
// POSIX Shared Memory for cross-process data transfer.

// Built this because I am hardware constrained and wanted to help my friends out.
// Alternate IPC methods like ProcessPoolExecutor pickle would kill our RAM budget.
// Took 4 attempts to map this memory block correctly without segfaulting Python.
"""

from multiprocessing import shared_memory
import struct
import json
import time
from typing import Optional

# Shared memory block size: 10MB - DO NOT CHANGE THIS UNLESS YOU WANT OUT OF MEMORY EXCEPTIONS
SHM_SIZE = 10 * 1024 * 1024  # 10MB
SHM_NAME = "sealmega_ipc"

# Header: 4 bytes status + 4 bytes data_length + 8 bytes timestamp = 16 bytes
HEADER_SIZE = 16
MAX_DATA_SIZE = SHM_SIZE - HEADER_SIZE

# Status codes
STATUS_EMPTY = 0
STATUS_WRITTEN = 1
STATUS_READING = 2
STATUS_PROCESSED = 3

_shm: Optional[shared_memory.SharedMemory] = None


def init_shared_memory() -> bool:
    """
    Allocate the 10MB shared memory block on boot.
    This is the ONLY data channel between processes.
    No pickle. No serialization queues. Pointers only.
    """
    global _shm
    
    try:
        # Try to attach to existing block
        _shm = shared_memory.SharedMemory(name=SHM_NAME)
        print(f"[IPC] Attached to existing shared memory block: {SHM_NAME} ({SHM_SIZE} bytes)")
    except FileNotFoundError:
        # Create new block
        _shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        # Zero out the header
        _shm.buf[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
        print(f"[IPC] Created shared memory block: {SHM_NAME} ({SHM_SIZE} bytes). God help us.")
    
    return True


def write_context(data: dict) -> int:
    """
    Write context data to shared memory.
    Returns the byte offset (pointer) where the data starts.
    
    The caller passes this integer to the model process.
    The model process reads directly from RAM. Zero copies. Python developers weep.
    """
    if _shm is None:
        raise RuntimeError("Shared memory not initialized. Call init_shared_memory() you idiot.")
    
    payload = json.dumps(data, ensure_ascii=True).encode('utf-8')
    
    if len(payload) > MAX_DATA_SIZE:
        # Truncate to fit — hard cap, no negotiation
        payload = payload[:MAX_DATA_SIZE]
    
    # Write header: status (4B) + data_length (4B) + timestamp (8B)
    now = time.time()
    header = struct.pack('<IId', STATUS_WRITTEN, len(payload), now)
    _shm.buf[:HEADER_SIZE] = header
    
    # Write payload directly to memory — zero copy from here
    _shm.buf[HEADER_SIZE:HEADER_SIZE + len(payload)] = payload
    
    return HEADER_SIZE  # Return the pointer offset


def read_context() -> Optional[dict]:
    """
    Read context from shared memory.
    Returns None if no data is available.
    
    The model process calls this. It reads directly from RAM.
    No deserialization queue. No pickle. 
    """
    if _shm is None:
        return None
    
    # Read header
    header = bytes(_shm.buf[:HEADER_SIZE])
    status, data_length, timestamp = struct.unpack('<IId', header)
    
    if status != STATUS_WRITTEN:
        return None
    
    if data_length == 0 or data_length > MAX_DATA_SIZE:
        return None
    
    # Mark as reading
    _shm.buf[:4] = struct.pack('<I', STATUS_READING)
    
    # Read payload directly from memory
    payload = bytes(_shm.buf[HEADER_SIZE:HEADER_SIZE + data_length])
    
    # Mark as processed
    _shm.buf[:4] = struct.pack('<I', STATUS_PROCESSED)
    
    try:
        return json.loads(payload.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def get_status() -> dict:
    """Get the current shared memory status."""
    if _shm is None:
        return {"initialized": False}
    
    header = bytes(_shm.buf[:HEADER_SIZE])
    status, data_length, timestamp = struct.unpack('<IId', header)
    
    status_names = {0: "empty", 1: "written", 2: "reading", 3: "processed"}
    
    return {
        "initialized": True,
        "name": SHM_NAME,
        "size_bytes": SHM_SIZE,
        "status": status_names.get(status, "unknown"),
        "data_length": data_length,
        "last_write": timestamp,
    }


def cleanup():
    """Release shared memory."""
    global _shm
    if _shm is not None:
        _shm.close()
        try:
            _shm.unlink()
        except Exception:
            pass
        _shm = None
        print("[IPC] Shared memory released.")
