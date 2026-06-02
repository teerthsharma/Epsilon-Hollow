/*
 * Ubuntu Benchmark Workloads — honest comparison with Seal OS benchmarks.
 *
 * Workloads mirror Seal OS kernel benchmarks:
 *   1. alloc-frame      → page allocation + touch + free
 *   2. mem-bandwidth    → sequential / random memory copy
 *   3. fs-teleport      → file rename (move) operations
 *   4. sched-yield      → thread ping-pong context switches
 *   5. tcp-demux        → hash-table TCP flow lookup simulation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>

#ifdef __x86_64__
#include <x86intrin.h>
static inline uint64_t rdtsc(void) { return __rdtsc(); }
#else
static inline uint64_t rdtsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static uint64_t percentile(uint64_t *samples, size_t n, int p) {
    if (n == 0) return 0;
    size_t idx = (n - 1) * p / 100;
    return samples[idx];
}

/* ── Workload 1: alloc-frame ────────────────────────────────────────────── */
void bench_alloc_frame(int iterations) {
    uint64_t *samples = calloc((size_t)iterations, sizeof(uint64_t));
    int ok = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        void *page = malloc(4096);
        if (page) {
            ((char *)page)[0] = (char)i;
            ((char *)page)[4095] = (char)(i * 31);
            free(page);
            ok++;
        }
        uint64_t end = rdtsc();
        samples[i] = end - start;
    }
    qsort(samples, (size_t)iterations, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = percentile(samples, (size_t)iterations, 50);
    uint64_t p95 = percentile(samples, (size_t)iterations, 95);
    uint64_t maxv = samples[iterations - 1];
    printf("[UBUNTU-BENCH] alloc-frame iterations=%d ok=%d bytes=4096 backend=malloc-page-touch-drop clock=rdtsc p50_cycles=%llu p95_cycles=%llu max_cycles=%llu\n",
           iterations, ok, (unsigned long long)p50, (unsigned long long)p95, (unsigned long long)maxv);
    free(samples);
}

/* ── Workload 2: mem-bandwidth ──────────────────────────────────────────── */
void bench_mem_bandwidth(int iterations) {
    size_t size = 64 * 1024 * 1024; /* 64 MB */
    char *src = malloc(size);
    char *dst = malloc(size);
    if (!src || !dst) {
        printf("[UBUNTU-BENCH] mem-bandwidth error=out_of_memory\n");
        free(src);
        free(dst);
        return;
    }
    memset(src, 0xAB, size);

    uint64_t *seq = calloc((size_t)iterations, sizeof(uint64_t));
    uint64_t *rnd = calloc((size_t)iterations, sizeof(uint64_t));

    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        memcpy(dst, src, size);
        uint64_t end = rdtsc();
        seq[i] = end - start;
    }

    int num_chunks = (int)(size / 4096);
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        for (int j = 0; j < 1000; j++) {
            size_t off = ((size_t)rand() % (size_t)num_chunks) * 4096;
            memcpy(dst + off, src + off, 4096);
        }
        uint64_t end = rdtsc();
        rnd[i] = end - start;
    }

    qsort(seq, (size_t)iterations, sizeof(uint64_t), cmp_u64);
    qsort(rnd, (size_t)iterations, sizeof(uint64_t), cmp_u64);

    printf("[UBUNTU-BENCH] mem-bandwidth iterations=%d size_mb=64 backend=memcpy clock=rdtsc seq_p50_cycles=%llu seq_p95_cycles=%llu rand_p50_cycles=%llu rand_p95_cycles=%llu\n",
           iterations,
           (unsigned long long)percentile(seq, (size_t)iterations, 50),
           (unsigned long long)percentile(seq, (size_t)iterations, 95),
           (unsigned long long)percentile(rnd, (size_t)iterations, 50),
           (unsigned long long)percentile(rnd, (size_t)iterations, 95));

    free(src);
    free(dst);
    free(seq);
    free(rnd);
}

/* ── Workload 3: fs-teleport ────────────────────────────────────────────── */
void bench_fs_teleport(int iterations) {
    const char *src_dir = "/tmp/teleport_src";
    const char *dst_dir = "/tmp/teleport_dst";
    mkdir(src_dir, 0755);
    mkdir(dst_dir, 0755);

    for (int i = 0; i < 100; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/file_%04d.txt", src_dir, i);
        int fd = open(path, O_WRONLY | O_CREAT, 0644);
        if (fd >= 0) {
            char buf[256];
            memset(buf, 'x', sizeof(buf));
            write(fd, buf, sizeof(buf));
            close(fd);
        }
    }

    uint64_t *samples = calloc((size_t)iterations, sizeof(uint64_t));
    int ok = 0;

    for (int i = 0; i < iterations; i++) {
        char src[256], dst[256];
        int idx = i % 100;
        snprintf(src, sizeof(src), "%s/file_%04d.txt", src_dir, idx);
        snprintf(dst, sizeof(dst), "%s/file_%04d.txt", dst_dir, idx);
        unlink(dst);

        uint64_t start = rdtsc();
        int r = rename(src, dst);
        uint64_t end = rdtsc();

        if (r == 0) ok++;
        samples[i] = end - start;

        rename(dst, src);
    }

    qsort(samples, (size_t)iterations, sizeof(uint64_t), cmp_u64);
    printf("[UBUNTU-BENCH] fs-teleport iterations=%d ok=%d files=100 backend=rename-tmpfs clock=rdtsc p50_cycles=%llu p95_cycles=%llu max_cycles=%llu\n",
           iterations, ok,
           (unsigned long long)percentile(samples, (size_t)iterations, 50),
           (unsigned long long)percentile(samples, (size_t)iterations, 95),
           (unsigned long long)samples[iterations - 1]);

    for (int i = 0; i < 100; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/file_%04d.txt", src_dir, i);
        unlink(path);
        snprintf(path, sizeof(path), "%s/file_%04d.txt", dst_dir, i);
        unlink(path);
    }
    rmdir(src_dir);
    rmdir(dst_dir);
    free(samples);
}

/* ── Workload 4: sched-yield ────────────────────────────────────────────── */
static volatile int sched_flag = 0;

static void *sched_thread(void *arg) {
    int me = *(int *)arg;
    int other = 1 - me;
    for (int i = 0; i < 1000; i++) {
        while (__sync_val_compare_and_swap(&sched_flag, other, me) != other) {
            /* spin */
        }
    }
    return NULL;
}

void bench_sched_yield(int iterations) {
    uint64_t *samples = calloc((size_t)iterations, sizeof(uint64_t));

    for (int i = 0; i < iterations; i++) {
        pthread_t t1, t2;
        int ids[2] = {0, 1};
        sched_flag = 0;

        uint64_t start = rdtsc();
        pthread_create(&t1, NULL, sched_thread, &ids[0]);
        pthread_create(&t2, NULL, sched_thread, &ids[1]);
        sched_flag = 1;
        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        uint64_t end = rdtsc();

        samples[i] = end - start;
    }

    qsort(samples, (size_t)iterations, sizeof(uint64_t), cmp_u64);
    printf("[UBUNTU-BENCH] sched-yield iterations=%d ping_pong=1000 backend=pthread-spin clock=rdtsc p50_cycles=%llu p95_cycles=%llu max_cycles=%llu\n",
           iterations,
           (unsigned long long)percentile(samples, (size_t)iterations, 50),
           (unsigned long long)percentile(samples, (size_t)iterations, 95),
           (unsigned long long)samples[iterations - 1]);
    free(samples);
}

/* ── Workload 5: tcp-demux ──────────────────────────────────────────────── */
#define TCP_FLOW_SLOTS 1024
#define TCP_FLOW_MASK  (TCP_FLOW_SLOTS - 1)

typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint32_t state;
} tcp_flow_t;

static tcp_flow_t tcp_flows[TCP_FLOW_SLOTS];

static uint32_t hash_flow(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port, uint16_t dst_port) {
    uint32_t h = src_ip ^ dst_ip ^ ((uint32_t)src_port << 16) ^ (uint32_t)dst_port;
    h = (h ^ (h >> 16)) * 0x45d9f3b;
    h = (h ^ (h >> 16)) * 0x45d9f3b;
    h = h ^ (h >> 16);
    return h & TCP_FLOW_MASK;
}

void bench_tcp_demux(int iterations) {
    memset(tcp_flows, 0, sizeof(tcp_flows));
    int num_flows = 512;
    for (int i = 0; i < num_flows; i++) {
        uint32_t slot = hash_flow((uint32_t)i, (uint32_t)(i + 1), (uint16_t)(10000 + i), 80);
        while (tcp_flows[slot].src_port != 0) {
            slot = (slot + 1) & TCP_FLOW_MASK;
        }
        tcp_flows[slot].src_ip = (uint32_t)i;
        tcp_flows[slot].dst_ip = (uint32_t)(i + 1);
        tcp_flows[slot].src_port = (uint16_t)(10000 + i);
        tcp_flows[slot].dst_port = 80;
        tcp_flows[slot].state = 1;
    }

    uint64_t *samples = calloc((size_t)iterations, sizeof(uint64_t));
    int lookups = 10000;
    int hits = 0;

    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        for (int j = 0; j < lookups; j++) {
            uint32_t f = (uint32_t)(j % num_flows);
            uint32_t slot = hash_flow(f, f + 1, (uint16_t)(10000 + f), 80);
            int probes = 0;
            while (tcp_flows[slot].src_port != 0) {
                probes++;
                if (tcp_flows[slot].src_ip == f &&
                    tcp_flows[slot].dst_ip == f + 1 &&
                    tcp_flows[slot].src_port == (uint16_t)(10000 + f) &&
                    tcp_flows[slot].dst_port == 80) {
                    hits++;
                    break;
                }
                slot = (slot + 1) & TCP_FLOW_MASK;
                if (probes > TCP_FLOW_SLOTS) break;
            }
        }
        uint64_t end = rdtsc();
        samples[i] = end - start;
    }

    qsort(samples, (size_t)iterations, sizeof(uint64_t), cmp_u64);
    printf("[UBUNTU-BENCH] tcp-demux iterations=%d lookups=%d flows=%d hits=%d backend=linear-probe-hash clock=rdtsc p50_cycles=%llu p95_cycles=%llu max_cycles=%llu\n",
           iterations, lookups, num_flows, hits,
           (unsigned long long)percentile(samples, (size_t)iterations, 50),
           (unsigned long long)percentile(samples, (size_t)iterations, 95),
           (unsigned long long)samples[iterations - 1]);
    free(samples);
}

/* ── Main ───────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <workload> <iterations>\n", argv[0]);
        fprintf(stderr, "Workloads: alloc-frame mem-bandwidth fs-teleport sched-yield tcp-demux\n");
        return 1;
    }
    const char *wl = argv[1];
    int iters = atoi(argv[2]);
    if (iters < 1) iters = 64;

    if (strcmp(wl, "alloc-frame") == 0) bench_alloc_frame(iters);
    else if (strcmp(wl, "mem-bandwidth") == 0) bench_mem_bandwidth(iters);
    else if (strcmp(wl, "fs-teleport") == 0) bench_fs_teleport(iters);
    else if (strcmp(wl, "sched-yield") == 0) bench_sched_yield(iters);
    else if (strcmp(wl, "tcp-demux") == 0) bench_tcp_demux(iters);
    else {
        fprintf(stderr, "Unknown workload: %s\n", wl);
        return 1;
    }
    return 0;
}
